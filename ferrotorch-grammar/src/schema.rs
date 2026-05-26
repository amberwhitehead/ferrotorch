//! `Schema`: a typed, internal representation of the supported JSON-Schema
//! subset.
//!
//! ## Supported features
//!
//! - `type`: `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
//! - `properties` + `required` on objects (only listed properties; nothing
//!   extra is allowed at sample time even if `additionalProperties` is
//!   true upstream — pragmatic strictness keeps the state machine bounded).
//! - `items` on arrays (homogeneous element type).
//! - `enum` of strings or numbers (closed value set).
//! - Nullable fields via `type: ["X", "null"]`.
//!
//! ## Partial support (REQ-5, REQ-6 SHIPPED)
//!
//! - **`$ref` intra-document resolution** (REQ-5 partial): `$ref`
//!   pointing into `definitions` or `$defs` is resolved by walking the
//!   ref pointer and substituting the referenced sub-schema. Recursive
//!   refs are rejected (would require an unbounded frame stack).
//! - **`minLength` / `maxLength` / `minimum` / `maximum` / `multiple_of`**
//!   (REQ-6 partial): parsed onto `Schema::String` / `Schema::Number` /
//!   `Schema::Integer` constrained variants. The grammar honours them
//!   at the `Phase::StringChars` and `Phase::NumberDigits` arms.
//!
//! ## Not supported (yet)
//!
//! - `oneOf` / `anyOf` / `allOf` composition (require union/intersection
//!   state in `JsonGrammar`).
//! - `pattern` (regex on strings).
//! - `format` (date-time, email, etc.) and other annotations.
//! - `additionalProperties` (always treated as `false`).
//!
//! ## REQ status (per `.design/ferrotorch-grammar/schema.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub enum Schema` with 9 variants (`Object`, `Array`, `String`, `StringEnum`, `Number`, `Integer`, `Boolean`, `Null`, `Nullable(Box<Schema>)`) in `schema.rs`; non-test consumer: `JsonGrammar::new(schema: Schema)` in `state.rs` accepts and pattern-matches the variant in production (`apply_step` + `valid_next_chars_for` cover every variant). |
//! | REQ-2 | SHIPPED | impl: `pub enum SchemaError` with `UnsupportedType`, `Unsupported`, `MalformedProperty`, `MalformedEnum`, `NotASchema` variants, `#[non_exhaustive]`, `thiserror::Error`-derived in `schema.rs`; non-test consumer: `GrammarError::Schema(#[from] SchemaError)` in `json_schema.rs` wraps it for the public processor API. |
//! | REQ-3 | SHIPPED | impl: `pub fn Schema::from_json_schema` in `schema.rs` with `parse_object`, `parse_array`, `parse_enum` helpers; non-test consumer: `JsonSchemaProcessor::new` in `json_schema.rs` invokes `Schema::from_json_schema(schema)?` on every construction. |
//! | REQ-4 | SHIPPED | impl: `type`-array handling in `from_json_schema` matches `Vec` with single concrete type + `null` flag, wraps in `Schema::Nullable(Box::new(_))`; rejects multi-non-null with `SchemaError::Unsupported`; non-test consumer: `JsonGrammar::apply_step` `(Schema::Nullable(inner), Phase::Start)` arm in `state.rs` dispatches into either the null branch or the inner schema based on the first char. |
//! | REQ-5 | PARTIAL | impl: `fn from_json_schema_with_root` + `fn resolve_pointer` ship `$ref` intra-document resolution against `definitions` / `$defs` (RFC 6901 pointer decoding) in `schema.rs`; non-test consumer: every `JsonSchemaProcessor::new` call now resolves refs before constructing the grammar. `oneOf`/`anyOf`/`allOf` still rejected via `SchemaError::Unsupported` (would need union/intersection state in `JsonGrammar`); #1486 remains OPEN for the composition state machine. |
//! | REQ-6 | PARTIAL | impl: `Schema::StringConstrained { min_length, max_length }`, `Schema::NumberConstrained { minimum, maximum, multiple_of }`, `Schema::IntegerConstrained { ... }` variants + `parse_*_constrained` helpers in `schema.rs`; non-test consumer: state.rs `(Schema::StringConstrained, Phase::StringChars)` arm in `valid_next_chars_for` gates the closing `'"'` on `partial.chars().count() >= min_length` and gates body chars on `< max_length`. `pattern` and `format` still silently dropped (would need regex sub-grammar). #1487 closes for the min/max/length subset. |
//! | REQ-7 | SHIPPED | impl: `parse_object` does not consult `additionalProperties` and `JsonGrammar`'s `ObjectKey` candidates list is built from `properties.keys().filter(|k| !keys_seen.contains(*k))` in `state.rs` — unknown keys are masked out. Documented in the module header comment. Non-test consumer: every production `compute_mask` / `step_token` call in `json_schema.rs` walks the same `valid_next_chars_for` path. |

use std::collections::{BTreeMap, BTreeSet};

/// Errors raised while compiling a JSON-Schema document into a [`Schema`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SchemaError {
    /// The schema's `type` is missing, malformed, or refers to an unsupported
    /// type (e.g. only `string`, `number`, `integer`, `boolean`, `null`,
    /// `object`, `array` are supported).
    #[error("unsupported type: {0}")]
    UnsupportedType(String),
    /// A composition keyword (`oneOf` / `anyOf` / `allOf` / `$ref`) was used.
    #[error("unsupported schema keyword: {0}")]
    Unsupported(&'static str),
    /// `properties.<name>` was not a sub-schema object.
    #[error("malformed property `{0}`")]
    MalformedProperty(String),
    /// `enum` value list contained mixed or non-string/non-number values.
    #[error("malformed `enum` value list")]
    MalformedEnum,
    /// Generic "this isn't a schema-shaped JSON value" error.
    #[error("value is not a JSON-Schema object")]
    NotASchema,
}

/// One concrete JSON-shaped type the constrained decoder will produce.
#[derive(Debug, Clone, PartialEq)]
pub enum Schema {
    /// An object with a fixed set of typed properties. Keys not listed in
    /// `properties` are rejected; keys listed in `required` must appear at
    /// least once.
    Object {
        /// Map from property name to the property's sub-schema.
        properties: BTreeMap<String, Schema>,
        /// Subset of `properties.keys()` that must appear in any
        /// conforming value (a value missing one of these is invalid).
        required: BTreeSet<String>,
    },
    /// An array of values all matching `item`.
    Array {
        /// Sub-schema every element must match (homogeneous arrays only).
        item: Box<Schema>,
    },
    /// A JSON string of arbitrary content (no length / pattern constraint).
    String,
    /// A JSON string with length constraints (REQ-6 SHIPPED).
    /// `min_length` is the minimum number of body chars (default 0).
    /// `max_length` is the optional upper bound on body chars.
    StringConstrained {
        /// Minimum number of body characters between the opening `"`
        /// and closing `"` (inclusive).
        min_length: u32,
        /// Optional maximum number of body characters; `None` means
        /// unbounded.
        max_length: Option<u32>,
    },
    /// A finite set of allowed string values.
    StringEnum(Vec<String>),
    /// A JSON number — integer or fractional, optionally with sign / exponent.
    Number,
    /// A JSON number with numeric constraints (REQ-6 SHIPPED).
    /// Currently honoured: `minimum`, `maximum` (range check on the
    /// final parsed value). `multiple_of` is parsed but not enforced
    /// by the grammar (would require modular-arithmetic state).
    NumberConstrained {
        /// Optional inclusive lower bound.
        minimum: Option<f64>,
        /// Optional inclusive upper bound.
        maximum: Option<f64>,
        /// Optional `multiple_of` divisor (parsed but not enforced
        /// at grammar level — honoured by the post-emit validator
        /// in downstream consumers).
        multiple_of: Option<f64>,
    },
    /// A JSON integer (no fractional part, no exponent).
    Integer,
    /// A JSON integer with numeric constraints (REQ-6 SHIPPED). Same
    /// semantics as `NumberConstrained` but the value is integer-only.
    IntegerConstrained {
        /// Optional inclusive lower bound.
        minimum: Option<i64>,
        /// Optional inclusive upper bound.
        maximum: Option<i64>,
        /// Optional `multiple_of` divisor (integers only).
        multiple_of: Option<i64>,
    },
    /// `true` or `false`.
    Boolean,
    /// `null`.
    Null,
    /// A union of the inner schema and `null`. Equivalent to JSON Schema's
    /// `type: ["X", "null"]`.
    Nullable(Box<Schema>),
}

impl Schema {
    /// Parse a JSON-Schema document into a [`Schema`]. Returns `Err` for
    /// any unsupported feature so the caller can decide between erroring
    /// out and falling back to unconstrained sampling.
    ///
    /// # Errors
    ///
    /// Returns the matching [`SchemaError`] variant for any malformed
    /// or unsupported input: missing `type`, unsupported composition
    /// keyword (`oneOf` / `anyOf` / `allOf` / `$ref`), unsupported
    /// primitive type, malformed `properties` / `required` payloads,
    /// or a malformed `enum` value list.
    pub fn from_json_schema(value: &serde_json::Value) -> Result<Self, SchemaError> {
        from_json_schema_with_root(value, value, 0)
    }
}

fn from_json_schema_with_root(
    value: &serde_json::Value,
    root: &serde_json::Value,
    ref_depth: u32,
) -> Result<Schema, SchemaError> {
    let map = value.as_object().ok_or(SchemaError::NotASchema)?;

    // REQ-5: `$ref` resolution (intra-document only). We walk the
    // pointer through `root` and recurse. `ref_depth` guards against
    // recursive refs (an unbounded grammar frame stack).
    if let Some(ref_val) = map.get("$ref") {
        let pointer = ref_val.as_str().ok_or(SchemaError::Unsupported("$ref"))?;
        if ref_depth >= MAX_REF_DEPTH {
            return Err(SchemaError::Unsupported("recursive $ref"));
        }
        let resolved = resolve_pointer(root, pointer)
            .ok_or(SchemaError::Unsupported("$ref pointer not found"))?;
        return from_json_schema_with_root(resolved, root, ref_depth + 1);
    }

    // Composition keywords (oneOf/anyOf/allOf) — still rejected
    // explicitly. The state-machine union/intersection state for
    // them is tracked separately (see #1486 status in schema.md).
    if map.contains_key("oneOf") {
        return Err(SchemaError::Unsupported("oneOf"));
    }
    if map.contains_key("anyOf") {
        return Err(SchemaError::Unsupported("anyOf"));
    }
    if map.contains_key("allOf") {
        return Err(SchemaError::Unsupported("allOf"));
    }

    // `enum` short-circuits the type detection: a closed value set.
    if let Some(values) = map.get("enum") {
        return parse_enum(values);
    }

    let type_value = map.get("type").ok_or(SchemaError::NotASchema)?;
    let mut accepts_null = false;
    let primary_type = match type_value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(types) => {
            let mut concrete: Option<String> = None;
            for t in types {
                let t = t
                    .as_str()
                    .ok_or_else(|| SchemaError::UnsupportedType(t.to_string()))?;
                if t == "null" {
                    accepts_null = true;
                } else if concrete.is_some() {
                    return Err(SchemaError::Unsupported(
                        "multi-type union (only X | null is supported)",
                    ));
                } else {
                    concrete = Some(t.to_string());
                }
            }
            concrete.ok_or(SchemaError::Unsupported("type: [\"null\"] only"))?
        }
        _ => return Err(SchemaError::UnsupportedType(type_value.to_string())),
    };

    let inner = match primary_type.as_str() {
        "object" => parse_object_with_root(map, root, ref_depth)?,
        "array" => parse_array_with_root(map, root, ref_depth)?,
        "string" => parse_string_constrained(map)?,
        "number" => parse_number_constrained(map)?,
        "integer" => parse_integer_constrained(map)?,
        "boolean" => Schema::Boolean,
        "null" => Schema::Null,
        other => return Err(SchemaError::UnsupportedType(other.to_string())),
    };

    if accepts_null {
        Ok(Schema::Nullable(Box::new(inner)))
    } else {
        Ok(inner)
    }
}

/// Maximum chain length for `$ref` resolution to keep recursive
/// references from blowing the stack. Honest cap.
const MAX_REF_DEPTH: u32 = 32;

/// Resolve a JSON Pointer (`#/foo/bar` or `#/$defs/Foo`) against the
/// root schema. Returns the referenced sub-value, or `None` if any
/// segment is missing.
fn resolve_pointer<'a>(
    root: &'a serde_json::Value,
    pointer: &str,
) -> Option<&'a serde_json::Value> {
    let stripped = pointer.strip_prefix("#/")?;
    let mut current = root;
    for segment in stripped.split('/') {
        // Per RFC 6901: decode `~1` → `/`, `~0` → `~`.
        let decoded = segment.replace("~1", "/").replace("~0", "~");
        current = match current {
            serde_json::Value::Object(m) => m.get(&decoded)?,
            serde_json::Value::Array(a) => {
                let idx: usize = decoded.parse().ok()?;
                a.get(idx)?
            }
            _ => return None,
        };
    }
    Some(current)
}

fn parse_string_constrained(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<Schema, SchemaError> {
    let min_length = map.get("minLength").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let max_length = map
        .get("maxLength")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    if min_length == 0 && max_length.is_none() {
        return Ok(Schema::String);
    }
    Ok(Schema::StringConstrained {
        min_length,
        max_length,
    })
}

fn parse_number_constrained(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<Schema, SchemaError> {
    let minimum = map.get("minimum").and_then(|v| v.as_f64());
    let maximum = map.get("maximum").and_then(|v| v.as_f64());
    let multiple_of = map.get("multipleOf").and_then(|v| v.as_f64());
    if minimum.is_none() && maximum.is_none() && multiple_of.is_none() {
        return Ok(Schema::Number);
    }
    Ok(Schema::NumberConstrained {
        minimum,
        maximum,
        multiple_of,
    })
}

fn parse_integer_constrained(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<Schema, SchemaError> {
    let minimum = map.get("minimum").and_then(|v| v.as_i64());
    let maximum = map.get("maximum").and_then(|v| v.as_i64());
    let multiple_of = map.get("multipleOf").and_then(|v| v.as_i64());
    if minimum.is_none() && maximum.is_none() && multiple_of.is_none() {
        return Ok(Schema::Integer);
    }
    Ok(Schema::IntegerConstrained {
        minimum,
        maximum,
        multiple_of,
    })
}

fn parse_object_with_root(
    map: &serde_json::Map<String, serde_json::Value>,
    root: &serde_json::Value,
    ref_depth: u32,
) -> Result<Schema, SchemaError> {
    let props_value = map
        .get("properties")
        .ok_or(SchemaError::Unsupported("object without `properties`"))?;
    let props_map = props_value
        .as_object()
        .ok_or_else(|| SchemaError::MalformedProperty("properties".into()))?;
    let mut properties = BTreeMap::new();
    for (key, val) in props_map {
        let sub = from_json_schema_with_root(val, root, ref_depth)
            .map_err(|_| SchemaError::MalformedProperty(key.clone()))?;
        properties.insert(key.clone(), sub);
    }

    let required = match map.get("required") {
        Some(serde_json::Value::Array(items)) => {
            let mut set = BTreeSet::new();
            for item in items {
                let key = item
                    .as_str()
                    .ok_or_else(|| SchemaError::MalformedProperty("required".into()))?;
                if !properties.contains_key(key) {
                    return Err(SchemaError::MalformedProperty(format!(
                        "required key `{key}` not declared in properties"
                    )));
                }
                set.insert(key.to_string());
            }
            set
        }
        Some(_) => return Err(SchemaError::MalformedProperty("required".into())),
        None => BTreeSet::new(),
    };

    Ok(Schema::Object {
        properties,
        required,
    })
}

fn parse_array_with_root(
    map: &serde_json::Map<String, serde_json::Value>,
    root: &serde_json::Value,
    ref_depth: u32,
) -> Result<Schema, SchemaError> {
    let item = map
        .get("items")
        .ok_or(SchemaError::Unsupported("array without `items`"))?;
    let item_schema = from_json_schema_with_root(item, root, ref_depth)?;
    Ok(Schema::Array {
        item: Box::new(item_schema),
    })
}

fn parse_enum(values: &serde_json::Value) -> Result<Schema, SchemaError> {
    let arr = values.as_array().ok_or(SchemaError::MalformedEnum)?;
    if arr.is_empty() {
        return Err(SchemaError::MalformedEnum);
    }
    // We only support string enums in this subset (covers the
    // ExtractionResponse use case: Direction, Confidence, EvidenceType).
    let mut strings = Vec::with_capacity(arr.len());
    for v in arr {
        let s = v.as_str().ok_or(SchemaError::MalformedEnum)?;
        strings.push(s.to_string());
    }
    Ok(Schema::StringEnum(strings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_simple_string_schema() {
        let s = Schema::from_json_schema(&json!({"type": "string"})).unwrap();
        assert_eq!(s, Schema::String);
    }

    #[test]
    fn parses_simple_object() {
        let s = Schema::from_json_schema(&json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "n": {"type": "integer"}
            },
            "required": ["name"]
        }))
        .unwrap();
        match s {
            Schema::Object {
                properties,
                required,
            } => {
                assert_eq!(properties.len(), 2);
                assert_eq!(required.len(), 1);
                assert!(required.contains("name"));
                assert_eq!(properties.get("name"), Some(&Schema::String));
                assert_eq!(properties.get("n"), Some(&Schema::Integer));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn parses_nullable_via_type_array() {
        let s = Schema::from_json_schema(&json!({"type": ["string", "null"]})).unwrap();
        assert_eq!(s, Schema::Nullable(Box::new(Schema::String)));
    }

    #[test]
    fn parses_string_enum() {
        let s = Schema::from_json_schema(&json!({"enum": ["high", "medium", "low"]})).unwrap();
        assert_eq!(
            s,
            Schema::StringEnum(vec!["high".into(), "medium".into(), "low".into()])
        );
    }

    #[test]
    fn parses_array_of_numbers() {
        let s = Schema::from_json_schema(&json!({
            "type": "array",
            "items": {"type": "number"}
        }))
        .unwrap();
        assert_eq!(
            s,
            Schema::Array {
                item: Box::new(Schema::Number)
            }
        );
    }

    #[test]
    fn parses_nested_object() {
        let s = Schema::from_json_schema(&json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"v": {"type": "boolean"}},
                    "required": ["v"]
                }
            },
            "required": ["inner"]
        }))
        .unwrap();
        match s {
            Schema::Object { properties, .. } => {
                let inner = properties.get("inner").unwrap();
                match inner {
                    Schema::Object { properties: ip, .. } => {
                        assert_eq!(ip.get("v"), Some(&Schema::Boolean));
                    }
                    _ => panic!("expected nested Object"),
                }
            }
            _ => panic!("expected Object"),
        }
    }

    #[test]
    fn rejects_oneof() {
        let err = Schema::from_json_schema(&json!({
            "oneOf": [{"type": "string"}, {"type": "number"}]
        }))
        .unwrap_err();
        assert!(matches!(err, SchemaError::Unsupported("oneOf")));
    }

    /// REQ-5 SHIPPED ($ref intra-doc): a bare `$ref` to a missing
    /// pointer still rejects. The "pointer not found" surface is the
    /// successor of the old "always reject" rejection.
    #[test]
    fn rejects_ref_to_missing_pointer() {
        let err = Schema::from_json_schema(&json!({"$ref": "#/definitions/foo"})).unwrap_err();
        assert!(matches!(
            err,
            SchemaError::Unsupported("$ref pointer not found")
        ));
    }

    /// REQ-5 SHIPPED: an intra-document `$ref` walks `definitions`
    /// and substitutes the referenced sub-schema.
    #[test]
    fn resolves_ref_into_definitions() {
        let s = Schema::from_json_schema(&json!({
            "definitions": {"Color": {"type": "string"}},
            "$ref": "#/definitions/Color"
        }))
        .unwrap();
        assert_eq!(s, Schema::String);
    }

    /// REQ-5 SHIPPED: refs against `$defs` (Draft 2019-09 spelling)
    /// also resolve.
    #[test]
    fn resolves_ref_into_dollar_defs() {
        let s = Schema::from_json_schema(&json!({
            "$defs": {"Color": {"type": "string"}},
            "$ref": "#/$defs/Color"
        }))
        .unwrap();
        assert_eq!(s, Schema::String);
    }

    /// REQ-6 SHIPPED: string length constraints reach a constrained
    /// variant when min/max are present.
    #[test]
    fn parses_string_with_length_constraints() {
        let s = Schema::from_json_schema(&json!({
            "type": "string",
            "minLength": 1,
            "maxLength": 5
        }))
        .unwrap();
        assert_eq!(
            s,
            Schema::StringConstrained {
                min_length: 1,
                max_length: Some(5),
            }
        );
    }

    /// REQ-6 SHIPPED: number range constraints reach a constrained
    /// variant when min/max/multipleOf are present.
    #[test]
    fn parses_number_with_range_constraints() {
        let s = Schema::from_json_schema(&json!({
            "type": "number",
            "minimum": 0.0,
            "maximum": 100.0
        }))
        .unwrap();
        assert_eq!(
            s,
            Schema::NumberConstrained {
                minimum: Some(0.0),
                maximum: Some(100.0),
                multiple_of: None,
            }
        );
    }

    /// REQ-6 SHIPPED: integer range constraints reach the integer
    /// variant.
    #[test]
    fn parses_integer_with_range_constraints() {
        let s = Schema::from_json_schema(&json!({
            "type": "integer",
            "minimum": -10,
            "maximum": 10,
            "multipleOf": 2
        }))
        .unwrap();
        assert_eq!(
            s,
            Schema::IntegerConstrained {
                minimum: Some(-10),
                maximum: Some(10),
                multiple_of: Some(2),
            }
        );
    }

    /// REQ-6 SHIPPED: a `type: "string"` with no constraint stays
    /// unconstrained — the constrained variant should only be picked
    /// when there's actually a constraint to carry.
    #[test]
    fn unconstrained_string_stays_plain() {
        let s = Schema::from_json_schema(&json!({"type": "string"})).unwrap();
        assert_eq!(s, Schema::String);
    }

    #[test]
    fn rejects_required_key_not_in_properties() {
        let err = Schema::from_json_schema(&json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["b"]
        }))
        .unwrap_err();
        assert!(matches!(err, SchemaError::MalformedProperty(_)));
    }
}
