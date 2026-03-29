//! # koe-asr
//!
//! Streaming ASR (Automatic Speech Recognition) client for Volcengine/Doubao, Qwen, and OpenAI Realtime.
//!
//! ## Quick Start (Doubao)
//!
//! ```rust,no_run
//! use koe_asr::{AsrConfig, AsrEvent, AsrProvider, DoubaoWsProvider, TranscriptAggregator};
//!
//! # async fn example() -> Result<(), koe_asr::AsrError> {
//! let config = AsrConfig {
//!     app_key: "your-app-key".into(),
//!     access_key: "your-access-key".into(),
//!     ..Default::default()
//! };
//!
//! let mut asr = DoubaoWsProvider::new();
//! asr.connect(&config).await?;
//!
//! // Push audio frames...
//! // asr.send_audio(&pcm_data).await?;
//! asr.finish_input().await?;
//!
//! let mut aggregator = TranscriptAggregator::new();
//! loop {
//!     match asr.next_event().await? {
//!         AsrEvent::Interim(text) => aggregator.update_interim(&text),
//!         AsrEvent::Definite(text) => aggregator.update_definite(&text),
//!         AsrEvent::Final(text) => { aggregator.update_final(&text); break; }
//!         AsrEvent::Closed => break,
//!         _ => {}
//!     }
//! }
//!
//! println!("{}", aggregator.best_text());
//! asr.close().await?;
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod doubao;
pub mod error;
pub mod event;
pub mod openai_realtime;
pub mod provider;
pub mod qwen;
pub mod transcript;

pub use config::{AsrConfig, OpenAiRealtimeBackend};
pub use openai_realtime::OpenAiRealtimeAsrProvider;
pub use qwen::QwenAsrProvider;
pub use doubao::DoubaoWsProvider;
pub use error::AsrError;
pub use event::AsrEvent;
pub use provider::AsrProvider;
pub use transcript::TranscriptAggregator;
