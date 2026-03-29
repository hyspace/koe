use crate::config::{AsrConfig, OpenAiRealtimeBackend};
use crate::error::{AsrError, Result};
use crate::event::AsrEvent;
use crate::provider::AsrProvider;
use base64::{engine::general_purpose::STANDARD, Engine};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::collections::VecDeque;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use uuid::Uuid;

type WsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

const TARGET_SAMPLE_RATE_HZ: u32 = 24_000;
const SESSION_EVENT_TIMEOUT: Duration = Duration::from_secs(10);

pub struct OpenAiRealtimeAsrProvider {
    ws: Option<WsStream>,
    pending_events: VecDeque<AsrEvent>,
    resampler: PairwiseUpsampler,
    input_finished: bool,
    pending_audio_since_commit: bool,
    final_emitted: bool,
    completed_text: Option<String>,
}

#[derive(Debug, Default)]
struct PairwiseUpsampler {
    carry_sample: Option<i16>,
}

impl PairwiseUpsampler {
    fn upsample_chunk(&mut self, input: &[u8]) -> Vec<u8> {
        let mut samples = Vec::with_capacity(input.len() / 2 + 1);

        if let Some(sample) = self.carry_sample.take() {
            samples.push(sample);
        }

        for chunk in input.chunks_exact(2) {
            samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }

        if samples.len() < 2 {
            self.carry_sample = samples.pop();
            return Vec::new();
        }

        let carry = if samples.len() % 2 == 1 {
            samples.pop()
        } else {
            None
        };

        let mut out = Vec::with_capacity(samples.len() * 3);
        for pair in samples.chunks_exact(2) {
            let left = pair[0];
            let right = pair[1];
            let mid = ((left as i32 + right as i32) / 2) as i16;
            out.extend_from_slice(&left.to_le_bytes());
            out.extend_from_slice(&mid.to_le_bytes());
            out.extend_from_slice(&right.to_le_bytes());
        }

        self.carry_sample = carry;
        out
    }

    fn flush(&mut self) -> Vec<u8> {
        if let Some(sample) = self.carry_sample.take() {
            let mut out = Vec::with_capacity(4);
            out.extend_from_slice(&sample.to_le_bytes());
            out.extend_from_slice(&sample.to_le_bytes());
            out
        } else {
            Vec::new()
        }
    }
}

impl OpenAiRealtimeAsrProvider {
    pub fn new() -> Self {
        Self {
            ws: None,
            pending_events: VecDeque::new(),
            resampler: PairwiseUpsampler::default(),
            input_finished: false,
            pending_audio_since_commit: false,
            final_emitted: false,
            completed_text: None,
        }
    }

    fn detect_backend(config: &AsrConfig) -> OpenAiRealtimeBackend {
        config
            .openai_realtime_backend
            .unwrap_or(OpenAiRealtimeBackend::OpenAi)
    }

    fn build_ws_url(config: &AsrConfig, backend: OpenAiRealtimeBackend) -> Result<String> {
        let base = config.url.trim();
        if base.is_empty() {
            return Err(AsrError::Connection("url is required".into()));
        }

        match backend {
            OpenAiRealtimeBackend::OpenAi => Ok(upsert_query_params(
                base,
                &[("intent", Some("transcription"))],
            )),
            OpenAiRealtimeBackend::AzureOpenAi => {
                let model = config.model.as_deref().unwrap_or("").trim();
                if model.is_empty() {
                    return Err(AsrError::Connection(
                        "model is required for Azure OpenAI realtime".into(),
                    ));
                }
                Ok(upsert_query_params(
                    base,
                    &[("deployment", Some(model)), ("intent", Some("transcription"))],
                ))
            }
        }
    }

    fn build_session_update(config: &AsrConfig) -> Value {
        let language = config.language.clone().unwrap_or_else(|| "zh".to_string());
        let prompt = config.prompt.clone().unwrap_or_default();
        let transcription_model = config
            .transcription_model
            .clone()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "gpt-4o-mini-transcribe".to_string());

        json!({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": TARGET_SAMPLE_RATE_HZ
                        },
                        "turn_detection": Value::Null,
                        "transcription": {
                            "model": transcription_model,
                            "language": language,
                            "prompt": prompt
                        }
                    }
                }
            }
        })
    }

    async fn send_json(&mut self, payload: Value) -> Result<()> {
        let text = serde_json::to_string(&payload)
            .map_err(|e| AsrError::Protocol(format!("serialize client event: {e}")))?;
        if let Some(ref mut ws) = self.ws {
            ws.send(Message::Text(text.into()))
                .await
                .map_err(|e| AsrError::Protocol(format!("send client event: {e}")))?;
            Ok(())
        } else {
            Err(AsrError::Connection("not connected".into()))
        }
    }

    fn parse_server_event(&mut self, value: Value) -> Vec<AsrEvent> {
        let event_type = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let mut events = Vec::new();

        match event_type {
            "session.created" => {}
            "session.updated" => events.push(AsrEvent::Connected),
            "input_audio_buffer.committed" => {
                self.pending_audio_since_commit = false;
            }
            "conversation.item.input_audio_transcription.completed" => {
                let transcript = value
                    .get("transcript")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !transcript.is_empty() {
                    self.completed_text = Some(transcript.clone());
                    events.push(AsrEvent::Definite(transcript.clone()));
                    if self.input_finished && !self.final_emitted {
                        self.final_emitted = true;
                        events.push(AsrEvent::Final(transcript));
                    }
                }
            }
            "conversation.item.input_audio_transcription.failed" | "error" => {
                events.push(AsrEvent::Error(extract_error_message(&value)));
            }
            _ => {}
        }

        events
    }

    fn maybe_queue_final(&mut self) {
        if self.final_emitted {
            return;
        }

        if let Some(text) = self.completed_text.clone() {
            self.pending_events.push_back(AsrEvent::Final(text));
            self.final_emitted = true;
        }
    }

    async fn read_text_event(ws: &mut WsStream, wait: Duration) -> Result<Value> {
        loop {
            let next = timeout(wait, ws.next())
                .await
                .map_err(|_| AsrError::Connection("timeout waiting for server event".into()))?;

            let message = match next {
                Some(Ok(message)) => message,
                Some(Err(e)) => return Err(AsrError::Connection(format!("WebSocket error: {e}"))),
                None => return Err(AsrError::Connection("connection closed".into())),
            };

            match message {
                Message::Text(text) => {
                    return serde_json::from_str(&text)
                        .map_err(|e| AsrError::Protocol(format!("parse server event: {e}")));
                }
                Message::Binary(data) => {
                    return serde_json::from_slice(&data).map_err(|e| {
                        AsrError::Protocol(format!("parse binary server event: {e}"))
                    });
                }
                Message::Close(frame) => {
                    return Err(AsrError::Connection(format!(
                        "connection closed unexpectedly: {:?}",
                        frame
                    )));
                }
                Message::Ping(_) | Message::Pong(_) => continue,
                other => {
                    return Err(AsrError::Protocol(format!(
                        "unexpected websocket message: {other:?}"
                    )));
                }
            }
        }
    }

    fn reset_runtime_state(&mut self) {
        self.pending_events.clear();
        self.resampler = PairwiseUpsampler::default();
        self.input_finished = false;
        self.pending_audio_since_commit = false;
        self.final_emitted = false;
        self.completed_text = None;
        self.ws = None;
    }
}

impl Default for OpenAiRealtimeAsrProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl AsrProvider for OpenAiRealtimeAsrProvider {
    async fn connect(&mut self, config: &AsrConfig) -> Result<()> {
        let api_key = config.access_key.clone();
        if api_key.is_empty() {
            return Err(AsrError::Connection("api_key is required".into()));
        }

        self.reset_runtime_state();

        let backend = Self::detect_backend(config);
        let ws_url = Self::build_ws_url(config, backend)?;

        let mut request = ws_url
            .into_client_request()
            .map_err(|e| AsrError::Connection(format!("invalid URL: {e}")))?;

        match backend {
            OpenAiRealtimeBackend::OpenAi => {
                request.headers_mut().insert(
                    "Authorization",
                    format!("Bearer {}", api_key)
                        .parse()
                        .map_err(|_| AsrError::Connection("invalid api_key".into()))?,
                );
            }
            OpenAiRealtimeBackend::AzureOpenAi => {
                request.headers_mut().insert(
                    "api-key",
                    api_key
                        .parse()
                        .map_err(|_| AsrError::Connection("invalid api_key".into()))?,
                );
            }
        }

        let (ws_stream, _) = timeout(Duration::from_millis(config.connect_timeout_ms), async {
            connect_async(request)
                .await
                .map_err(|e| AsrError::Connection(e.to_string()))
        })
        .await
        .map_err(|_| AsrError::Connection("connection timed out".into()))??;

        self.ws = Some(ws_stream);

        if let Some(ref mut ws) = self.ws {
            let created = Self::read_text_event(ws, SESSION_EVENT_TIMEOUT).await?;
            let created_type = created
                .get("type")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            if created_type != "session.created" {
                return Err(AsrError::Connection(format!(
                    "expected session.created event, got {created_type}"
                )));
            }
            let events = self.parse_server_event(created);
            self.pending_events.extend(events);
        }

        self.send_json(Self::build_session_update(config)).await?;

        loop {
            match self.next_event().await? {
                AsrEvent::Connected => return Ok(()),
                AsrEvent::Error(msg) => return Err(AsrError::Protocol(msg)),
                AsrEvent::Closed => {
                    return Err(AsrError::Connection(
                        "connection closed before session.updated".into(),
                    ));
                }
                AsrEvent::Interim(_) | AsrEvent::Definite(_) | AsrEvent::Final(_) => {}
            }
        }
    }

    async fn send_audio(&mut self, frame: &[u8]) -> Result<()> {
        if frame.is_empty() {
            return Ok(());
        }

        let upsampled = self.resampler.upsample_chunk(frame);
        if upsampled.is_empty() {
            return Ok(());
        }

        self.pending_audio_since_commit = true;
        self.send_json(json!({
            "event_id": format!("event_{}", Uuid::new_v4()),
            "type": "input_audio_buffer.append",
            "audio": STANDARD.encode(upsampled),
        }))
        .await
    }

    async fn finish_input(&mut self) -> Result<()> {
        if self.input_finished {
            return Ok(());
        }

        self.input_finished = true;

        let flushed = self.resampler.flush();
        if !flushed.is_empty() {
            self.pending_audio_since_commit = true;
            self.send_json(json!({
                "event_id": format!("event_{}", Uuid::new_v4()),
                "type": "input_audio_buffer.append",
                "audio": STANDARD.encode(flushed),
            }))
            .await?;
        }

        if self.pending_audio_since_commit {
            self.send_json(json!({
                "event_id": format!("event_{}", Uuid::new_v4()),
                "type": "input_audio_buffer.commit"
            }))
            .await?;
        } else {
            self.maybe_queue_final();
        }

        Ok(())
    }

    async fn next_event(&mut self) -> Result<AsrEvent> {
        loop {
            if let Some(event) = self.pending_events.pop_front() {
                return Ok(event);
            }

            if let Some(ref mut ws) = self.ws {
                match ws.next().await {
                    Some(Ok(Message::Text(text))) => {
                        let value: Value = serde_json::from_str(&text)
                            .map_err(|e| AsrError::Protocol(format!("parse server event: {e}")))?;
                        let events = self.parse_server_event(value);
                        self.pending_events.extend(events);
                    }
                    Some(Ok(Message::Binary(data))) => {
                        let value: Value = serde_json::from_slice(&data).map_err(|e| {
                            AsrError::Protocol(format!("parse binary server event: {e}"))
                        })?;
                        let events = self.parse_server_event(value);
                        self.pending_events.extend(events);
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        self.maybe_queue_final();
                        if let Some(event) = self.pending_events.pop_front() {
                            return Ok(event);
                        }
                        return Ok(AsrEvent::Closed);
                    }
                    Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => continue,
                    Some(Ok(_)) => continue,
                    Some(Err(e)) => return Err(AsrError::Protocol(e.to_string())),
                }
            } else {
                return Err(AsrError::Connection("not connected".into()));
            }
        }
    }

    async fn close(&mut self) -> Result<()> {
        self.maybe_queue_final();
        if let Some(mut ws) = self.ws.take() {
            let _ = ws.close(None).await;
        }
        Ok(())
    }
}

fn extract_error_message(value: &Value) -> String {
    value
        .get("error")
        .and_then(|err| err.get("message"))
        .and_then(|msg| msg.as_str())
        .or_else(|| value.get("message").and_then(|msg| msg.as_str()))
        .unwrap_or("Unknown error")
        .to_string()
}

fn upsert_query_params(url: &str, updates: &[(&str, Option<&str>)]) -> String {
    let mut parts = url.splitn(2, '?');
    let path = parts.next().unwrap_or(url);
    let query = parts.next().unwrap_or("");

    let mut pairs: Vec<(String, String)> = query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .filter_map(|pair| {
            let mut pair_parts = pair.splitn(2, '=');
            Some((
                pair_parts.next()?.to_string(),
                pair_parts.next().unwrap_or("").to_string(),
            ))
        })
        .collect();

    for (key, value) in updates {
        if let Some(index) = pairs.iter().position(|(existing_key, _)| existing_key == key) {
            match value {
                Some(value) => pairs[index].1 = (*value).to_string(),
                None => {
                    pairs.remove(index);
                }
            }
        } else if let Some(value) = value {
            pairs.push(((*key).to_string(), (*value).to_string()));
        }
    }

    if pairs.is_empty() {
        path.to_string()
    } else {
        let query = pairs
            .into_iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join("&");
        format!("{path}?{query}")
    }
}
