//! A GGUF file's header (version 2 and 3, little-endian), for src/ippon/gguf/reader.py: the metadata and the tensor
//! table as JSON that Python's json module reads back to the values Python's struct module gives. Integers stay
//! integers, a float32 becomes the float64 it is exactly, floats keep a decimal point or exponent (so 1000000.0 is
//! not read back as an int), NaN and infinities are written as Python's json writes them, and a bool is any nonzero
//! byte. With `skip_tokenizer`, arrays under "tokenizer.ggml." are stepped over without being decoded: ippon reads the
//! creator's tokenizer.json, and in a Qwen-vocabulary file those arrays are some 744,000 values (106 ms in Python on an
//! M5 Pro for Clef-Flash's Q8_0 file, read three times a load).

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, Read};

const STRING: u32 = 8;
const ARRAY: u32 = 9;
// how deep arrays of arrays may go: GGUF's own files nest one deep, and each level is a call on the stack, which a
// crafted file nesting thousands deep would overflow (an abort, which no catch_unwind stops)
const MAX_DEPTH: usize = 4;

struct Reader {
    inner: BufReader<File>,
    at: u64,
    len: u64, // the file's size: nothing the header says may be read past it
}

impl Reader {
    /// The next n bytes. A size the file can't hold is refused before anything is allocated for it: the header's own
    /// counts and lengths say how much to read, and a crafted one could ask for more memory than the Mac has.
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        if n as u64 > self.len.saturating_sub(self.at) {
            return Err("the header runs past the end of the file".to_string());
        }
        let mut out = vec![0u8; n];
        self.inner.read_exact(&mut out).map_err(|_| "the header runs past the end of the file".to_string())?;
        self.at += n as u64;
        Ok(out)
    }

    /// Steps over n bytes; reading past the end shows at the next read, or in the tensors' offsets.
    fn skip(&mut self, n: u64) -> Result<(), String> {
        if n > self.len.saturating_sub(self.at) {
            return Err("the header runs past the end of the file".to_string());
        }
        self.inner.seek_relative(n as i64).map_err(|e| e.to_string())?;
        self.at += n;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()?;
        String::from_utf8(self.bytes(n as usize)?).map_err(|_| "a metadata string is not UTF-8".into())
    }

    /// One value of type `kind`, written to `out` as JSON, or stepped over when `out` is None.
    fn value(&mut self, kind: u32, out: Option<&mut String>) -> Result<(), String> {
        self.nested(kind, out, 0)
    }

    fn nested(&mut self, kind: u32, out: Option<&mut String>, depth: usize) -> Result<(), String> {
        let width = match kind {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            STRING => {
                return match out {
                    None => {
                        let n = self.u64()?;
                        self.skip(n)
                    }
                    Some(out) => {
                        let text = self.string()?;
                        quote(out, &text);
                        Ok(())
                    }
                };
            }
            ARRAY => {
                if depth >= MAX_DEPTH {
                    return Err(format!("GGUF metadata arrays nested deeper than {MAX_DEPTH}"));
                }
                let inner = self.u32()?;
                let n = self.u64()?;
                // every element takes a byte or more: a count the rest of the file can't hold is a crafted one, and
                // stepping through it would never end
                if n > self.len.saturating_sub(self.at) {
                    return Err("the header runs past the end of the file".to_string());
                }
                return match out {
                    None => (0..n).try_for_each(|_| self.nested(inner, None, depth + 1)),
                    Some(out) => {
                        out.push('[');
                        for i in 0..n {
                            if i > 0 {
                                out.push(',');
                            }
                            self.nested(inner, Some(out), depth + 1)?;
                        }
                        out.push(']');
                        Ok(())
                    }
                };
            }
            _ => return Err(format!("a GGUF metadata value of unknown type {kind}")),
        };
        let raw = self.bytes(width)?;
        let Some(out) = out else { return Ok(()) };
        match kind {
            0 => write!(out, "{}", raw[0]),
            1 => write!(out, "{}", raw[0] as i8),
            2 => write!(out, "{}", u16::from_le_bytes([raw[0], raw[1]])),
            3 => write!(out, "{}", i16::from_le_bytes([raw[0], raw[1]])),
            4 => write!(out, "{}", u32::from_le_bytes(raw[..4].try_into().unwrap())),
            5 => write!(out, "{}", i32::from_le_bytes(raw[..4].try_into().unwrap())),
            6 => {
                float(out, f32::from_le_bytes(raw[..4].try_into().unwrap()) as f64);
                Ok(())
            }
            7 => write!(out, "{}", if raw[0] != 0 { "true" } else { "false" }),
            10 => write!(out, "{}", u64::from_le_bytes(raw[..8].try_into().unwrap())),
            11 => write!(out, "{}", i64::from_le_bytes(raw[..8].try_into().unwrap())),
            _ => {
                float(out, f64::from_le_bytes(raw[..8].try_into().unwrap()));
                Ok(())
            }
        }
        .map_err(|e| e.to_string())
    }
}

fn float(out: &mut String, x: f64) {
    if x.is_nan() {
        out.push_str("NaN");
    } else if x.is_infinite() {
        out.push_str(if x > 0.0 { "Infinity" } else { "-Infinity" });
    } else {
        let _ = write!(out, "{x:?}"); // the shortest form that reads back to x, with a "." or an exponent
    }
}

fn quote(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The header as JSON: {"version", "end" (the header's length in bytes), "metadata" {key: value} in file order,
/// "tensors" [[name, ggml type, dims innermost first, offset]]}, or {"error": why}.
pub fn header(path: &str, skip_tokenizer: bool) -> String {
    match read(path, skip_tokenizer) {
        Ok(json) => json,
        Err(why) => {
            let mut out = String::from("{\"error\":");
            quote(&mut out, &why);
            out.push('}');
            out
        }
    }
}

fn read(path: &str, skip_tokenizer: bool) -> Result<String, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    let mut r = Reader { inner: BufReader::with_capacity(1 << 20, file), at: 0, len };
    if r.bytes(4)? != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        return Err(format!("GGUF version {version} (ippon reads 2 and 3)"));
    }
    let (count, kv) = (r.u64()?, r.u64()?);
    let mut out = format!("{{\"version\":{version},\"metadata\":{{");
    let mut first = true;
    for _ in 0..kv {
        let key = r.string()?;
        let kind = r.u32()?;
        if skip_tokenizer && kind == ARRAY && key.starts_with("tokenizer.ggml.") {
            r.value(kind, None)?;
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        quote(&mut out, &key);
        out.push(':');
        r.value(kind, Some(&mut out))?;
    }
    out.push_str("},\"tensors\":[");
    for i in 0..count {
        if i > 0 {
            out.push(',');
        }
        let name = r.string()?;
        let dims = r.u32()?;
        let shape = (0..dims).map(|_| r.u64().map(|d| d.to_string())).collect::<Result<Vec<_>, _>>()?;
        let (kind, offset) = (r.u32()?, r.u64()?);
        out.push('[');
        quote(&mut out, &name);
        let _ = write!(out, ",{kind},[{}],{offset}]", shape.join(","));
    }
    let _ = write!(out, "],\"end\":{}}}", r.at);
    Ok(out)
}
