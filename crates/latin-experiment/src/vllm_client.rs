//! vLLM client for OpenAI-compatible chat completions API.
//!
//! Replaces ollama-rs with direct HTTP calls to vLLM server.
//! Supports system prompts via the chat completions API.
//!
//! For multi-model setups (model escalation), supports routing requests
//! to different vLLM instances based on model name.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Map model names to vLLM server ports for multi-model setups.
///
/// When running multiple vLLM instances (one per model), each instance
/// listens on a different port. This function maps model names to ports:
/// - Port 8001: 0.5B models
/// - Port 8002: 1.5B models
/// - Port 8003: 3B models
/// - Port 8004: 7B models
/// - Port 8005: 14B models
///
/// Returns None if model routing is not applicable (single-model setup).
fn model_to_port(model: &str) -> Option<u16> {
    // Only route if model name contains size identifier
    if model.contains("0.5B") || model.contains("0.5b") {
        Some(8001)
    } else if model.contains("1.5B") || model.contains("1.5b") {
        Some(8002)
    } else if model.contains("3B") || model.contains("3b") {
        Some(8003)
    } else if model.contains("7B") || model.contains("7b") {
        Some(8004)
    } else if model.contains("14B") || model.contains("14b") {
        Some(8005)
    } else {
        None // Use base_url as-is
    }
}

/// Default system prompt for Latin Square solving (replaces Ollama Modelfile).
pub const LATIN_SYSTEM_PROMPT: &str = "You solve Latin Square puzzles. Given a row with \
    empty cells (_), return ONLY the number(s) that fill them. Return just the numbers, \
    nothing else.";

/// vLLM client for chat completions.
#[derive(Clone)]
pub struct VllmClient {
    client: reqwest::Client,
    base_url: String,
}

/// A chat message with role and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Request body for /v1/chat/completions.
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
}

/// Response from /v1/chat/completions.
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

/// A single choice in the response.
#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

impl VllmClient {
    /// Create a new vLLM client.
    ///
    /// # Arguments
    /// * `base_url` - The base URL of the vLLM server (e.g., "http://localhost:8000")
    pub fn new(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Generate a response using the default Latin Square system prompt.
    ///
    /// # Arguments
    /// * `model` - Model name (e.g., "Qwen/Qwen2.5-0.5B")
    /// * `prompt` - User prompt
    /// * `temperature` - Sampling temperature (0.0-1.0)
    /// * `top_p` - Nucleus sampling parameter (0.0-1.0)
    /// * `max_tokens` - Maximum tokens to generate
    pub async fn generate(
        &self,
        model: &str,
        prompt: &str,
        temperature: f32,
        top_p: f32,
        max_tokens: u32,
    ) -> Result<String> {
        self.generate_with_system(model, LATIN_SYSTEM_PROMPT, prompt, temperature, top_p, max_tokens)
            .await
    }

    /// Generate a response with a custom system prompt.
    ///
    /// Used by conversation.rs for agent-specific system prompts.
    pub async fn generate_with_system(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        temperature: f32,
        top_p: f32,
        max_tokens: u32,
    ) -> Result<String> {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            },
        ];

        let request = ChatRequest {
            model: model.to_string(),
            messages,
            max_tokens,
            temperature,
            top_p,
        };

        // Route to correct vLLM instance based on model name (for multi-model setups)
        let url = if let Some(port) = model_to_port(model) {
            // Multi-model setup: route to specific port based on model size
            format!("http://localhost:{}/v1/chat/completions", port)
        } else {
            // Single-model setup: use configured base_url
            format!("{}/v1/chat/completions", self.base_url)
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request to vLLM server")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("vLLM request failed with status {}: {}", status, body);
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .context("Failed to parse vLLM response")?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .context("No choices in vLLM response")
    }

    /// Check if the vLLM server is healthy.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);
        match self.client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = VllmClient::new("http://localhost:8000");
        assert_eq!(client.base_url, "http://localhost:8000");

        // Test trailing slash removal
        let client = VllmClient::new("http://localhost:8000/");
        assert_eq!(client.base_url, "http://localhost:8000");
    }

    #[test]
    fn test_system_prompt_constant() {
        assert!(LATIN_SYSTEM_PROMPT.contains("Latin Square"));
        assert!(LATIN_SYSTEM_PROMPT.contains("empty cells"));
    }

    #[test]
    fn test_model_to_port() {
        // Test HuggingFace format model names
        assert_eq!(model_to_port("Qwen/Qwen2.5-0.5B"), Some(8001));
        assert_eq!(model_to_port("Qwen/Qwen2.5-1.5B"), Some(8002));
        assert_eq!(model_to_port("Qwen/Qwen2.5-3B"), Some(8003));
        assert_eq!(model_to_port("Qwen/Qwen2.5-7B"), Some(8004));
        assert_eq!(model_to_port("Qwen/Qwen2.5-14B"), Some(8005));

        // Test case insensitivity
        assert_eq!(model_to_port("qwen2.5-0.5b"), Some(8001));
        assert_eq!(model_to_port("model-7b-instruct"), Some(8004));

        // Test fallback for unknown model sizes
        assert_eq!(model_to_port("some-unknown-model"), None);
        assert_eq!(model_to_port("custom-model"), None);
    }
}
