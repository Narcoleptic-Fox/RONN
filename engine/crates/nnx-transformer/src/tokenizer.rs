//! BPE tokenizer — encode text to token IDs and decode back.
//!
//! Reads vocabulary and merge rules from GGUF metadata. Supports the
//! standard BPE algorithm used by Llama, Mistral, and most modern LLMs.

use std::collections::HashMap;

/// A loaded BPE tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Token ID → string piece.
    vocab: Vec<String>,
    /// String piece → token ID.
    token_to_id: HashMap<String, u32>,
    /// Merge rules: (piece_a, piece_b) → merged piece, ordered by priority.
    merges: Vec<(String, String)>,
    /// Special token IDs.
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub unknown_token_id: u32,
}

impl Tokenizer {
    /// Build a tokenizer from vocabulary and merge rules.
    pub fn new(
        vocab: Vec<String>,
        merges: Vec<(String, String)>,
        bos_token_id: u32,
        eos_token_id: u32,
    ) -> Self {
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        // Try to find <unk> token
        let unknown_token_id = token_to_id.get("<unk>").copied().unwrap_or(0);

        Self {
            vocab,
            token_to_id,
            merges,
            bos_token_id,
            eos_token_id,
            unknown_token_id,
        }
    }

    /// Build a tokenizer from GGUF metadata.
    pub fn from_gguf(metadata: &nnx_gguf::GGUFMetadata) -> Result<Self, String> {
        // Extract vocabulary
        let tokens_val = metadata
            .get("tokenizer.ggml.tokens")
            .ok_or("missing tokenizer.ggml.tokens")?;

        let vocab: Vec<String> = match tokens_val {
            nnx_gguf::types::GGUFValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    nnx_gguf::types::GGUFValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => return Err("tokenizer.ggml.tokens is not an array".into()),
        };

        // Extract merges (optional — some tokenizers don't have explicit merges)
        let merges = if let Some(merges_val) = metadata.get("tokenizer.ggml.merges") {
            match merges_val {
                nnx_gguf::types::GGUFValue::Array(arr) => arr
                    .iter()
                    .filter_map(|v| match v {
                        nnx_gguf::types::GGUFValue::String(s) => {
                            let parts: Vec<&str> = s.splitn(2, ' ').collect();
                            if parts.len() == 2 {
                                Some((parts[0].to_string(), parts[1].to_string()))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };

        // Special tokens
        let bos_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);
        let eos_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);

        Ok(Self::new(vocab, merges, bos_id, eos_id))
    }

    /// Encode text into token IDs.
    ///
    /// Uses BPE: start with characters, iteratively merge the highest-priority pair.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with individual bytes/characters as pieces
        let mut pieces: Vec<String> = text.bytes().map(|b| self.byte_to_piece(b)).collect();

        // Build merge priority lookup (owned keys for lifetime safety)
        let merge_priority: HashMap<(String, String), usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();

        // Iteratively apply the highest-priority merge
        loop {
            if pieces.len() < 2 {
                break;
            }

            // Find the merge with the lowest index (highest priority)
            let mut best_idx = None;
            let mut best_priority = usize::MAX;

            for i in 0..pieces.len() - 1 {
                let key = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&priority) = merge_priority.get(&key) {
                    if priority < best_priority {
                        best_priority = priority;
                        best_idx = Some(i);
                    }
                }
            }

            match best_idx {
                Some(idx) => {
                    let merged = format!("{}{}", pieces[idx], pieces[idx + 1]);
                    pieces[idx] = merged;
                    pieces.remove(idx + 1);
                }
                None => break, // No more merges possible
            }
        }

        // Convert pieces to token IDs
        pieces
            .iter()
            .map(|piece| {
                self.token_to_id
                    .get(piece)
                    .copied()
                    .unwrap_or(self.unknown_token_id)
            })
            .collect()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            let piece = self.id_to_piece(id);
            // Handle byte-level tokens like <0x0A> for newline
            if let Some(byte_val) = Self::parse_byte_token(&piece) {
                bytes.push(byte_val);
            } else {
                bytes.extend_from_slice(piece.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Get the string piece for a token ID.
    pub fn id_to_piece(&self, id: u32) -> &str {
        self.vocab
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>")
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Convert a byte to its BPE piece representation.
    /// Llama uses a byte-level BPE where each byte maps to a unicode character.
    fn byte_to_piece(&self, byte: u8) -> String {
        // First check if the byte token exists in vocab as <0xXX>
        let hex_token = format!("<0x{:02X}>", byte);
        if self.token_to_id.contains_key(&hex_token) {
            return hex_token;
        }
        // Otherwise try the character directly
        let ch = byte as char;
        if self.token_to_id.contains_key(&ch.to_string()) {
            return ch.to_string();
        }
        // Fallback to the hex representation
        hex_token
    }

    /// Parse a byte-level token like "<0x0A>" back to its byte value.
    fn parse_byte_token(piece: &str) -> Option<u8> {
        if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
            u8::from_str_radix(&piece[3..5], 16).ok()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_tokenizer() -> Tokenizer {
        let vocab = vec![
            "<unk>".into(),
            "<s>".into(),
            "</s>".into(),
            "h".into(),
            "e".into(),
            "l".into(),
            "o".into(),
            "he".into(),
            "ll".into(),
            "lo".into(),
            "hel".into(),
            "hello".into(),
        ];
        // Merge priority matters: l+o must come before l+l so "hello" can fully merge.
        // h,e,l,l,o → he,l,l,o → he,l,lo → hel,lo → hello
        let merges = vec![
            ("h".into(), "e".into()),    // 0: h + e → he
            ("l".into(), "o".into()),    // 1: l + o → lo (before l+l!)
            ("he".into(), "l".into()),   // 2: he + l → hel
            ("hel".into(), "lo".into()), // 3: hel + lo → hello
        ];
        Tokenizer::new(vocab, merges, 1, 2)
    }

    #[test]
    fn test_encode_with_merges() {
        let tok = simple_tokenizer();
        let ids = tok.encode("hello");
        // h,e,l,l,o → (h+e)=he → he,l,l,o → (l+o)=lo → he,l,lo
        // → (he+l)=hel → hel,lo → (hel+lo)=hello → token 11
        assert_eq!(ids, vec![11]);
    }

    #[test]
    fn test_decode() {
        let tok = simple_tokenizer();
        assert_eq!(tok.decode(&[11]), "hello");
        assert_eq!(tok.decode(&[3, 4]), "he");
    }

    #[test]
    fn test_unknown_token() {
        let tok = simple_tokenizer();
        // 'x' is not in vocab, should produce <unk>
        let ids = tok.encode("x");
        assert_eq!(ids, vec![0]); // <unk> is ID 0
    }
}
