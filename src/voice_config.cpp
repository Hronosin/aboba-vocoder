// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/voice_config.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace aboba {

namespace {

// ============================================================
// Minimal TOML subset parser.
//
// We deliberately keep this self-contained. Audio frameworks shouldn't
// drag in 30KLOC of dependencies for a config file. The cost is that we
// don't support the full TOML 1.0.0 spec — only what voice configs need.
// ============================================================

struct ParseError {
    std::string message;
    int line = 0;
};

// Trim leading/trailing whitespace
std::string trim(std::string s) {
    auto not_space = [](int c) { return !std::isspace(c); };
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(),
            s.end());
    return s;
}

std::string strip_comment(const std::string& s) {
    // Find # outside of a string literal. Strings start/end at unescaped ".
    bool in_string = false;
    bool escape    = false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (escape) { escape = false; continue; }
        if (c == '\\' && in_string) { escape = true; continue; }
        if (c == '"') { in_string = !in_string; continue; }
        if (c == '#' && !in_string) {
            return s.substr(0, i);
        }
    }
    return s;
}

// Parse a TOML string literal: "...". Returns the unescaped content.
// Caller must verify s starts with "; returns content + sets `end_pos` to
// the index AFTER the closing ".
std::string parse_string(const std::string& s, std::size_t start,
                         std::size_t& end_pos, ParseError* err) {
    std::string result;
    if (start >= s.size() || s[start] != '"') {
        if (err) err->message = "expected '\"' at start of string";
        return {};
    }
    std::size_t i = start + 1;
    while (i < s.size()) {
        char c = s[i];
        if (c == '"') {
            end_pos = i + 1;
            return result;
        }
        if (c == '\\' && i + 1 < s.size()) {
            char nxt = s[i + 1];
            switch (nxt) {
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case '\\': result += '\\'; break;
                case '"':  result += '"';  break;
                default:
                    if (err) err->message = std::string("unknown escape \\") + nxt;
                    return {};
            }
            i += 2;
            continue;
        }
        result += c;
        ++i;
    }
    if (err) err->message = "unterminated string";
    return {};
}

// Try to interpret a raw value string as: bool / int / float / string.
// On success, fills exactly ONE of the out parameters. Returns true.
bool parse_value(const std::string& raw_in,
                 bool* out_bool, long* out_int, double* out_float,
                 std::string* out_str, ParseError* err) {
    std::string raw = trim(raw_in);
    if (raw.empty()) {
        if (err) err->message = "empty value";
        return false;
    }

    // String
    if (raw[0] == '"') {
        std::size_t end;
        ParseError sub;
        std::string s = parse_string(raw, 0, end, &sub);
        if (!sub.message.empty()) {
            if (err) *err = sub;
            return false;
        }
        // Anything (non-whitespace) after the closing quote = error.
        if (end != raw.size() &&
            trim(raw.substr(end)).size() != 0) {
            if (err) err->message = "trailing junk after string";
            return false;
        }
        *out_str = s;
        return true;
    }

    // Bool
    if (raw == "true")  { *out_bool = true;  return true; }
    if (raw == "false") { *out_bool = false; return true; }

    // Numeric: float if it contains a '.', 'e', or 'E'
    bool is_float = (raw.find('.') != std::string::npos)
                  || (raw.find('e') != std::string::npos)
                  || (raw.find('E') != std::string::npos);

    char* endp = nullptr;
    if (is_float) {
        const double d = std::strtod(raw.c_str(), &endp);
        if (endp == raw.c_str() || *endp != '\0') {
            if (err) err->message = "invalid number: '" + raw + "'";
            return false;
        }
        *out_float = d;
        return true;
    } else {
        const long n = std::strtol(raw.c_str(), &endp, 10);
        if (endp == raw.c_str() || *endp != '\0') {
            if (err) err->message = "invalid integer: '" + raw + "'";
            return false;
        }
        *out_int = n;
        return true;
    }
}

// A single (section, key) -> string-raw value. We re-parse the raw value
// at access time to give the requested type, so type errors are
// attributable to the key, not the section.
struct Entry {
    std::string raw;
    int         line = 0;
};
using Section = std::map<std::string, Entry>;
using SectionMap = std::map<std::string, Section>;

// Parse `text` into sections. Returns true / false; on false fills `err`.
bool parse_toml(const std::string& text, SectionMap& out, ParseError& err) {
    std::string current_section = "";   // "" = top-level
    std::stringstream ss(text);
    std::string raw_line;
    int line_no = 0;

    while (std::getline(ss, raw_line)) {
        ++line_no;
        std::string line = trim(strip_comment(raw_line));
        if (line.empty()) continue;

        // Section header: [name]
        if (line.front() == '[') {
            if (line.back() != ']') {
                err.message = "unclosed section header";
                err.line = line_no;
                return false;
            }
            std::string name = trim(line.substr(1, line.size() - 2));
            if (name.empty()) {
                err.message = "empty section name";
                err.line = line_no;
                return false;
            }
            // Reject unusual characters — allow only alnum + '_' + '.' + '-'
            for (char c : name) {
                if (!std::isalnum(static_cast<unsigned char>(c)) &&
                    c != '_' && c != '.' && c != '-') {
                    err.message = "invalid character in section name: '"
                                 + std::string(1, c) + "'";
                    err.line = line_no;
                    return false;
                }
            }
            current_section = name;
            // Touch the section so empty sections still exist.
            out[name];
            continue;
        }

        // key = value
        auto eq = line.find('=');
        if (eq == std::string::npos) {
            err.message = "expected 'key = value'";
            err.line = line_no;
            return false;
        }
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (key.empty()) {
            err.message = "empty key";
            err.line = line_no;
            return false;
        }
        for (char c : key) {
            if (!std::isalnum(static_cast<unsigned char>(c)) &&
                c != '_' && c != '-') {
                err.message = "invalid character in key: '"
                             + std::string(1, c) + "'";
                err.line = line_no;
                return false;
            }
        }
        // Note: we DON'T validate `val` here. The type-conversion step
        // will catch malformed values when we read them.
        out[current_section][key] = Entry{val, line_no};
    }
    return true;
}

// Accessor helpers. They return whether the key was present, NOT whether
// the parsed value was valid — type validation happens inside.
bool get_string(const Section& s, const char* key, std::string& out,
                ParseError& err) {
    auto it = s.find(key);
    if (it == s.end()) return false;
    bool b; long n; double d; std::string str;
    ParseError sub;
    if (!parse_value(it->second.raw, &b, &n, &d, &str, &sub)) {
        err = sub; err.line = it->second.line; return false;
    }
    if (sub.message.empty() && !str.empty()) {
        out = str; return true;
    }
    // If parse succeeded but it wasn't a string, it's a type mismatch.
    err.message = std::string("'") + key + "' must be a string";
    err.line = it->second.line;
    return false;
}

bool get_bool(const Section& s, const char* key, bool& out, ParseError& err) {
    auto it = s.find(key);
    if (it == s.end()) return false;
    bool b = false; long n; double d; std::string str;
    ParseError sub;
    if (!parse_value(it->second.raw, &b, &n, &d, &str, &sub)) {
        err = sub; err.line = it->second.line; return false;
    }
    // Disambiguate: parse_value sets exactly one. We test for boolean here
    // by re-checking the raw value (true/false are unambiguous tokens).
    const std::string trimmed = trim(it->second.raw);
    if (trimmed == "true" || trimmed == "false") {
        out = (trimmed == "true");
        return true;
    }
    err.message = std::string("'") + key + "' must be a bool";
    err.line = it->second.line;
    return false;
}

bool get_int(const Section& s, const char* key, long& out, ParseError& err) {
    auto it = s.find(key);
    if (it == s.end()) return false;
    bool b; long n = 0; double d; std::string str;
    ParseError sub;
    if (!parse_value(it->second.raw, &b, &n, &d, &str, &sub)) {
        err = sub; err.line = it->second.line; return false;
    }
    // If raw contained a dot/e/E it would be float; otherwise int.
    const std::string trimmed = trim(it->second.raw);
    if (trimmed.find('.') == std::string::npos &&
        trimmed.find('e') == std::string::npos &&
        trimmed.find('E') == std::string::npos &&
        trimmed[0] != '"') {
        out = n;
        return true;
    }
    err.message = std::string("'") + key + "' must be an integer";
    err.line = it->second.line;
    return false;
}

bool get_float(const Section& s, const char* key, double& out, ParseError& err) {
    auto it = s.find(key);
    if (it == s.end()) return false;
    bool b; long n; double d = 0.0; std::string str;
    ParseError sub;
    if (!parse_value(it->second.raw, &b, &n, &d, &str, &sub)) {
        err = sub; err.line = it->second.line; return false;
    }
    const std::string trimmed = trim(it->second.raw);
    if (trimmed[0] == '"') {
        err.message = std::string("'") + key + "' must be a number";
        err.line = it->second.line;
        return false;
    }
    if (trimmed.find('.') == std::string::npos &&
        trimmed.find('e') == std::string::npos &&
        trimmed.find('E') == std::string::npos) {
        out = static_cast<double>(n);  // accept ints as floats
    } else {
        out = d;
    }
    return true;
}

// Convert string to QualityProfile.
bool parse_profile(const std::string& s, QualityProfile& out) {
    if (s == "quality")     { out = QualityProfile::Quality;     return true; }
    if (s == "balanced")    { out = QualityProfile::Balanced;    return true; }
    if (s == "performance") { out = QualityProfile::Performance; return true; }
    return false;
}

// Convert string to MusicalScale.
bool parse_scale(const std::string& s, MusicalScale& out) {
    if (s == "chromatic")        { out = MusicalScale::Chromatic;       return true; }
    if (s == "major")            { out = MusicalScale::Major;           return true; }
    if (s == "minor")            { out = MusicalScale::Minor;           return true; }
    if (s == "harmonic-minor" ||
        s == "harmonic_minor")   { out = MusicalScale::HarmonicMinor;   return true; }
    if (s == "pentatonic-major" ||
        s == "pentatonic_major") { out = MusicalScale::PentatonicMajor; return true; }
    if (s == "pentatonic-minor" ||
        s == "pentatonic_minor") { out = MusicalScale::PentatonicMinor; return true; }
    if (s == "blues")            { out = MusicalScale::Blues;           return true; }
    if (s == "whole-tone" ||
        s == "whole_tone")       { out = MusicalScale::WholeTone;       return true; }
    return false;
}

// Parse note letter "C" "C#" "Db" ... to 0..11
bool parse_root(const std::string& s, int& out) {
    if (s.empty()) return false;
    // Numeric form
    if (std::isdigit(static_cast<unsigned char>(s[0]))) {
        char* endp; long n = std::strtol(s.c_str(), &endp, 10);
        if (*endp != '\0' || n < 0 || n > 11) return false;
        out = static_cast<int>(n);
        return true;
    }
    // Letter form
    char up = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    int base = -1;
    switch (up) {
        case 'C': base = 0;  break;
        case 'D': base = 2;  break;
        case 'E': base = 4;  break;
        case 'F': base = 5;  break;
        case 'G': base = 7;  break;
        case 'A': base = 9;  break;
        case 'B': base = 11; break;
        default: return false;
    }
    int accidental = 0;
    if (s.size() >= 2) {
        if (s[1] == '#') accidental = +1;
        else if (s[1] == 'b') accidental = -1;
        else return false;
    }
    int n = (base + accidental) % 12;
    if (n < 0) n += 12;
    out = n;
    return true;
}

}  // namespace

// ============================================================
// VoiceConfig methods
// ============================================================

VoicePipelineConfig VoiceConfig::to_pipeline_config() const {
    VoicePipelineConfig pc;
    pc.sample_rate = sample_rate;
    pc.fft_size    = fft_size;
    pc.hop_size    = hop_size;
    pc.profile     = profile;
    pc.enable_gate          = noise_gate;
    pc.enable_highpass      = highpass;
    pc.highpass_cutoff_hz   = highpass_cutoff_hz;
    pc.enable_noise_reducer = noise_reducer;
    pc.enable_agc           = agc;
    pc.enable_de_esser      = de_esser;
    pc.enable_reverb        = reverb;
    pc.enable_autotune      = autotune_enabled;
    pc.use_lookahead_limiter = lookahead;
    return pc;
}

void VoiceConfig::apply_runtime(VoicePipeline& pipe) const {
    if (has_character) {
        pipe.set_character(character);
    } else {
        pipe.set_pitch_semitones(pitch_semitones);
        pipe.set_formant_semitones(formant_semitones);
    }
    if (autotune_enabled) {
        pipe.set_autotune_enabled(true);
        pipe.set_autotune_scale(autotune_scale, autotune_root);
        pipe.set_autotune_strength(autotune_strength);
        pipe.set_autotune_glide_ms(autotune_glide_ms);
    } else {
        pipe.set_autotune_enabled(false);
    }
    if (reverb) {
        pipe.set_reverb_room_size(reverb_room_size);
        pipe.set_reverb_damping(reverb_damping);
        pipe.set_reverb_wet(reverb_wet);
    }
}

// ============================================================
// Loaders
// ============================================================

VoiceConfigResult parse_voice_config(const std::string& toml_text,
                                     const std::string& source_label) {
    (void)source_label;
    VoiceConfigResult result;

    // Paranoia: same cap as load_voice_config. Catches in-process callers
    // that pass adversarial input directly.
    constexpr std::size_t kMaxConfigBytes = 1 * 1024 * 1024;
    if (toml_text.size() > kMaxConfigBytes) {
        result.error = "config text too large (>" +
            std::to_string(kMaxConfigBytes) + " bytes)";
        result.line_number = 0;
        return result;
    }

    SectionMap sections;
    ParseError perr;
    if (!parse_toml(toml_text, sections, perr)) {
        result.error = "parse error: " + perr.message;
        result.line_number = perr.line;
        return result;
    }

    auto cfg = std::make_unique<VoiceConfig>();
    ParseError sub;

    // Top-level
    auto& root = sections[""];
    {
        std::string s;
        if (get_string(root, "name", s, sub))         cfg->name = s;
        else if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
        if (get_string(root, "description", s, sub))  cfg->description = s;
        else if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
    }

    // [pipeline]
    if (sections.count("pipeline")) {
        auto& p = sections["pipeline"];
        std::string s; long n; double d;
        if (get_string(p, "profile", s, sub)) {
            if (!parse_profile(s, cfg->profile)) {
                result.error = "unknown profile: '" + s + "'";
                result.line_number = p["profile"].line;
                return result;
            }
        } else if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
        if (get_float(p, "sample_rate", d, sub))  cfg->sample_rate = d;
        if (get_int  (p, "fft_size",   n, sub))   cfg->fft_size = static_cast<std::size_t>(n);
        if (get_int  (p, "hop_size",   n, sub))   cfg->hop_size = static_cast<std::size_t>(n);
        if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
    }

    // [pitch]
    if (sections.count("pitch")) {
        auto& p = sections["pitch"];
        double d;
        if (get_float(p, "semitones",         d, sub)) cfg->pitch_semitones = static_cast<float>(d);
        if (get_float(p, "formant_semitones", d, sub)) cfg->formant_semitones = static_cast<float>(d);
        if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
    }

    // [character]
    if (sections.count("character")) {
        auto& p = sections["character"];
        std::string s;
        if (get_string(p, "preset", s, sub)) {
            auto c = character_from_id(s.c_str());
            if (c == VoiceCharacter::Count) {
                result.error = "unknown character preset: '" + s + "'";
                result.line_number = p["preset"].line;
                return result;
            }
            cfg->character = c;
            cfg->has_character = true;
        } else if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
    }

    // [autotune]
    if (sections.count("autotune")) {
        auto& p = sections["autotune"];
        std::string s; bool b; double d;
        if (get_bool  (p, "enabled",   b, sub)) cfg->autotune_enabled = b;
        if (get_string(p, "scale",     s, sub)) {
            if (!parse_scale(s, cfg->autotune_scale)) {
                result.error = "unknown scale: '" + s + "'";
                result.line_number = p["scale"].line;
                return result;
            }
        } else if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
        // root: accept either string ("A", "C#", "Db") or integer (0..11)
        if (auto it_root = p.find("root"); it_root != p.end()) {
            const std::string& raw = trim(it_root->second.raw);
            if (!raw.empty() && raw[0] == '"') {
                // String form
                std::string s_root;
                ParseError sub_root;
                if (!get_string(p, "root", s_root, sub_root)) {
                    if (!sub_root.message.empty()) {
                        result.error = sub_root.message;
                        result.line_number = sub_root.line;
                        return result;
                    }
                }
                if (!parse_root(s_root, cfg->autotune_root)) {
                    result.error = "unknown root note: '" + s_root + "'";
                    result.line_number = it_root->second.line;
                    return result;
                }
            } else {
                // Integer form
                long n_root = 0;
                ParseError sub_root;
                if (!get_int(p, "root", n_root, sub_root)) {
                    result.error = sub_root.message.empty()
                        ? "'root' must be string or int 0..11"
                        : sub_root.message;
                    result.line_number = it_root->second.line;
                    return result;
                }
                if (n_root < 0 || n_root > 11) {
                    result.error = "root must be 0..11";
                    result.line_number = it_root->second.line;
                    return result;
                }
                cfg->autotune_root = static_cast<int>(n_root);
            }
        }
        if (get_float(p, "strength", d, sub)) cfg->autotune_strength = static_cast<float>(d);
        if (get_float(p, "glide_ms", d, sub)) cfg->autotune_glide_ms = static_cast<float>(d);
        if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
    }

    // [effects]
    if (sections.count("effects")) {
        auto& p = sections["effects"];
        bool b; double d;
        if (get_bool (p, "noise_gate",         b, sub)) cfg->noise_gate    = b;
        if (get_bool (p, "highpass",           b, sub)) cfg->highpass      = b;
        if (get_float(p, "highpass_cutoff_hz", d, sub)) cfg->highpass_cutoff_hz = static_cast<float>(d);
        if (get_bool (p, "noise_reducer",      b, sub)) cfg->noise_reducer = b;
        if (get_bool (p, "agc",                b, sub)) cfg->agc           = b;
        if (get_bool (p, "de_esser",           b, sub)) cfg->de_esser      = b;
        if (get_bool (p, "reverb",             b, sub)) cfg->reverb        = b;
        if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
    }

    // [reverb]
    if (sections.count("reverb")) {
        auto& p = sections["reverb"];
        double d;
        if (get_float(p, "room_size", d, sub)) cfg->reverb_room_size = static_cast<float>(d);
        if (get_float(p, "damping",   d, sub)) cfg->reverb_damping   = static_cast<float>(d);
        if (get_float(p, "wet",       d, sub)) cfg->reverb_wet       = static_cast<float>(d);
        if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
    }

    // [limiter]
    if (sections.count("limiter")) {
        auto& p = sections["limiter"];
        bool b;
        if (get_bool(p, "lookahead", b, sub)) cfg->lookahead = b;
        if (!sub.message.empty()) { result.error = sub.message; result.line_number = sub.line; return result; }
    }

    result.value = std::move(cfg);
    return result;
}

VoiceConfigResult load_voice_config(const std::string& path) {
    VoiceConfigResult result;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        result.error = "cannot open file: " + path;
        return result;
    }
    // Paranoia: cap file size. Sane TOML configs are well under 100 KiB.
    // A 1 MiB cap leaves a generous safety margin without exposing us to
    // gigabyte-sized adversarial input.
    constexpr std::streamsize kMaxConfigBytes = 1 * 1024 * 1024;
    f.seekg(0, std::ios::end);
    const auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz < 0) {
        result.error = "cannot determine file size: " + path;
        return result;
    }
    if (sz > kMaxConfigBytes) {
        result.error = "config file too large (>" +
            std::to_string(kMaxConfigBytes) + " bytes): " + path;
        return result;
    }
    std::string buf;
    buf.resize(static_cast<std::size_t>(sz));
    f.read(buf.data(), sz);
    return parse_voice_config(buf, path);
}

std::string serialize_voice_config(const VoiceConfig& cfg) {
    std::stringstream s;
    s << "# Aboba voice config\n";
    s << "name = \"" << cfg.name << "\"\n";
    if (!cfg.description.empty()) {
        s << "description = \"" << cfg.description << "\"\n";
    }
    s << "\n[pipeline]\n";
    s << "profile = \"" << profile_name(cfg.profile) << "\"\n";
    s << "sample_rate = " << cfg.sample_rate << "\n";
    s << "fft_size = " << cfg.fft_size << "\n";
    s << "hop_size = " << cfg.hop_size << "\n";
    s << "\n[pitch]\n";
    s << "semitones = " << cfg.pitch_semitones << "\n";
    s << "formant_semitones = " << cfg.formant_semitones << "\n";
    if (cfg.has_character) {
        s << "\n[character]\n";
        s << "preset = \"" << character_id(cfg.character) << "\"\n";
    }
    s << "\n[autotune]\n";
    s << "enabled = " << (cfg.autotune_enabled ? "true" : "false") << "\n";
    s << "scale = \"" << scale_name(cfg.autotune_scale) << "\"\n";
    s << "root = " << cfg.autotune_root << "\n";
    s << "strength = " << cfg.autotune_strength << "\n";
    s << "glide_ms = " << cfg.autotune_glide_ms << "\n";
    s << "\n[effects]\n";
    s << "noise_gate = "    << (cfg.noise_gate ? "true" : "false") << "\n";
    s << "highpass = "      << (cfg.highpass ? "true" : "false") << "\n";
    s << "highpass_cutoff_hz = " << cfg.highpass_cutoff_hz << "\n";
    s << "noise_reducer = " << (cfg.noise_reducer ? "true" : "false") << "\n";
    s << "agc = "           << (cfg.agc ? "true" : "false") << "\n";
    s << "de_esser = "      << (cfg.de_esser ? "true" : "false") << "\n";
    s << "reverb = "        << (cfg.reverb ? "true" : "false") << "\n";
    if (cfg.reverb) {
        s << "\n[reverb]\n";
        s << "room_size = " << cfg.reverb_room_size << "\n";
        s << "damping = "   << cfg.reverb_damping   << "\n";
        s << "wet = "       << cfg.reverb_wet       << "\n";
    }
    s << "\n[limiter]\n";
    s << "lookahead = " << (cfg.lookahead ? "true" : "false") << "\n";
    return s.str();
}

}  // namespace aboba
