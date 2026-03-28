// Copyright 2026 Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Text chunking strategies for streaming TTS.
//!
//! The streaming chunker aggressively splits the first chunk at the earliest
//! clause boundary for minimal time-to-first-audio, then uses sentence-level
//! splitting for subsequent chunks.

/// Default maximum length for the first (aggressive) chunk.
pub const FIRST_CHUNK_MAX: usize = 60;

/// Default maximum length for subsequent chunks.
pub const REST_CHUNK_MAX: usize = 400;

/// Chunk text for streaming TTS.
///
/// The first chunk is split aggressively at clause boundaries (`,;:—-`) to
/// minimize latency to first audio. Subsequent chunks are split at sentence
/// boundaries (`[.!?]`).
pub fn chunk_text_streaming(text: &str, first_max: usize, rest_max: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    // If text is short enough, return as single chunk
    if text.len() <= first_max {
        return vec![ensure_punctuation(text)];
    }

    let mut chunks = Vec::new();

    // First chunk: split at earliest clause boundary for minimal latency
    let first_chunk_end = find_clause_boundary(text, first_max);
    let (first, rest) = text.split_at(first_chunk_end);
    let first = first.trim();
    if !first.is_empty() {
        chunks.push(ensure_punctuation(first));
    }

    // Remaining text: split at sentence boundaries
    let rest = rest.trim();
    if !rest.is_empty() {
        let sentence_chunks = chunk_by_sentences(rest, rest_max);
        chunks.extend(sentence_chunks);
    }

    chunks
}

/// Find the best clause boundary position within `max_len` characters.
///
/// Looks for `,`, `;`, `:`, `—`, `-` boundaries. Falls back to word boundary,
/// then to `max_len`.
fn find_clause_boundary(text: &str, max_len: usize) -> usize {
    let search_end = max_len.min(text.len());
    let search_text = &text[..search_end];

    // Look for clause-level punctuation (last occurrence within limit)
    let clause_delimiters = [',', ';', ':', '\u{2014}' /* em-dash */];
    let mut best_pos = None;
    for (i, c) in search_text.char_indices() {
        if clause_delimiters.contains(&c) {
            best_pos = Some(i + c.len_utf8());
        }
    }
    if let Some(pos) = best_pos {
        return pos;
    }

    // Fall back to word boundary
    if let Some(pos) = search_text.rfind(' ') {
        return pos + 1;
    }

    // Last resort: split at max_len
    search_end
}

/// Split text into chunks at sentence boundaries.
///
/// Each sentence becomes its own chunk for streaming (one TTS generation per
/// sentence). Only sentences that exceed `max_len` are split further at word
/// boundaries.
fn chunk_by_sentences(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    // Split on sentence-ending punctuation
    let sentences = split_sentences(text);

    for sentence in sentences {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        if sentence.len() <= max_len {
            chunks.push(ensure_punctuation(sentence));
        } else {
            // Long sentence: split at word boundaries
            let mut remaining = sentence.to_string();
            while remaining.len() > max_len {
                let split_pos = find_word_boundary(&remaining, max_len);
                let (left, right) = remaining.split_at(split_pos);
                chunks.push(ensure_punctuation(left.trim()));
                remaining = right.trim().to_string();
            }
            if !remaining.is_empty() {
                chunks.push(ensure_punctuation(&remaining));
            }
        }
    }

    chunks
}

/// Split text into sentences at `.`, `!`, `?` boundaries.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;

    for (i, c) in text.char_indices() {
        if c == '.' || c == '!' || c == '?' {
            let end = i + c.len_utf8();
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            start = end;
        }
    }

    // Remaining text after last sentence-ending punctuation
    let remaining = text[start..].trim();
    if !remaining.is_empty() {
        sentences.push(remaining.to_string());
    }

    sentences
}

/// Find a word boundary position near `max_len`.
fn find_word_boundary(text: &str, max_len: usize) -> usize {
    let search_end = max_len.min(text.len());
    let search_text = &text[..search_end];
    if let Some(pos) = search_text.rfind(' ') {
        pos + 1
    } else {
        search_end
    }
}

/// Ensure text ends with punctuation. Appends a comma if none found.
fn ensure_punctuation(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return text.to_string();
    }
    let last = text.chars().last().unwrap();
    if ".!?,;:".contains(last) {
        text.to_string()
    } else {
        format!("{},", text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_single_chunk() {
        let chunks = chunk_text_streaming("Hello world!", 60, 400);
        assert_eq!(chunks, vec!["Hello world!"]);
    }

    #[test]
    fn test_aggressive_first_chunk() {
        let text = "Hello there, this is a much longer piece of text that should be split into multiple chunks for streaming.";
        let chunks = chunk_text_streaming(text, 60, 400);
        assert!(chunks.len() >= 2);
        // First chunk should split at a clause boundary
        assert!(chunks[0].len() <= 60);
    }

    #[test]
    fn test_three_chunks_comma_then_sentences() {
        let text = "Hello there, this is a streaming test. The quick brown fox jumps over the lazy dog.";
        let chunks = chunk_text_streaming(text, 60, 400);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "Hello there,");
        assert_eq!(chunks[1], "this is a streaming test.");
        assert_eq!(chunks[2], "The quick brown fox jumps over the lazy dog.");
    }

    #[test]
    fn test_sentence_splitting() {
        let text = "First sentence. Second sentence. Third sentence here that is also present. And a fourth one too.";
        let chunks = chunk_text_streaming(text, 20, 400);
        // First chunk at word boundary within 20 chars, then each sentence is its own chunk
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn test_ensure_punctuation() {
        assert_eq!(ensure_punctuation("hello"), "hello,");
        assert_eq!(ensure_punctuation("hello."), "hello.");
        assert_eq!(ensure_punctuation("hello!"), "hello!");
        assert_eq!(ensure_punctuation("hello,"), "hello,");
    }

    #[test]
    fn test_empty_text() {
        let chunks = chunk_text_streaming("", 60, 400);
        assert!(chunks.is_empty());
    }
}
