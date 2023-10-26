from __future__ import annotations

import gradio as gr
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import sys
import os
base_path = os.path.dirname(sys.path[0])
sys.path.append(os.path.join(base_path, "src"))
print(sys.path)
from seamless_communication.models.inference.translator import Translator

DESCRIPTION = """# SeamlessM4T

[SeamlessM4T](https://github.com/facebookresearch/seamless_communication) is designed to provide high-quality
translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.

This unified model enables multiple tasks like Speech-to-Speech (S2ST), Speech-to-Text (S2TT), Text-to-Speech (T2ST)
translation and more, without relying on multiple separate models.
"""

TASK_NAMES = [
    "S2ST (Speech to Speech translation)",
    "S2TT (Speech to Text translation)",
    "T2ST (Text to Speech translation)",
    "T2TT (Text to Text translation)",
    "ASR (Automatic Speech Recognition)",
]

# Language dict
language_code_to_name = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arb": "Modern Standard Arabic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "North Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "fra": "French",
    "gaz": "West Central Oromo",
    "gle": "Irish",
    "glg": "Galician",
    "guj": "Gujarati",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kam": "Kamba",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kea": "Kabuverdianu",
    "khk": "Halh Mongolian",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "luo": "Luo",
    "lvs": "Standard Latvian",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mni": "Meitei",
    "mya": "Burmese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokm\u00e5l",
    "npi": "Nepali",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "pan": "Punjabi",
    "pbt": "Southern Pashto",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Cantonese",
    "zlm": "Colloquial Malay",
    "zsm": "Standard Malay",
    "zul": "Zulu",
}
LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}

# Source langs: S2ST / S2TT / ASR don't need source lang
# T2TT / T2ST use this
text_source_language_codes = [
    "afr",
    "amh",
    "arb",
    "ary",
    "arz",
    "asm",
    "azj",
    "bel",
    "ben",
    "bos",
    "bul",
    "cat",
    "ceb",
    "ces",
    "ckb",
    "cmn",
    "cym",
    "dan",
    "deu",
    "ell",
    "eng",
    "est",
    "eus",
    "fin",
    "fra",
    "gaz",
    "gle",
    "glg",
    "guj",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "ibo",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kan",
    "kat",
    "kaz",
    "khk",
    "khm",
    "kir",
    "kor",
    "lao",
    "lit",
    "lug",
    "luo",
    "lvs",
    "mai",
    "mal",
    "mar",
    "mkd",
    "mlt",
    "mni",
    "mya",
    "nld",
    "nno",
    "nob",
    "npi",
    "nya",
    "ory",
    "pan",
    "pbt",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "spa",
    "srp",
    "swe",
    "swh",
    "tam",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
    "yor",
    "yue",
    "zsm",
    "zul",
]
TEXT_SOURCE_LANGUAGE_NAMES = sorted(
    [language_code_to_name[code] for code in text_source_language_codes]
)

# Target langs:
# S2ST / T2ST
s2st_target_language_codes = [
    "eng",
    "arb",
    "ben",
    "cat",
    "ces",
    "cmn",
    "cym",
    "dan",
    "deu",
    "est",
    "fin",
    "fra",
    "hin",
    "ind",
    "ita",
    "jpn",
    "kor",
    "mlt",
    "nld",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "spa",
    "swe",
    "swh",
    "tel",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
]
S2ST_TARGET_LANGUAGE_NAMES = sorted(
    [language_code_to_name[code] for code in s2st_target_language_codes]
)
# S2TT / ASR
S2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
# T2TT
T2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES

# Download sample input audio files
filenames = ["assets/sample_input.mp3", "assets/sample_input_2.mp3"]
for filename in filenames:
    hf_hub_download(
        repo_id="facebook/seamless_m4t",
        repo_type="space",
        filename=filename,
        local_dir=".",
    )

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 120  # in seconds
DEFAULT_TARGET_LANGUAGE = "French"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
translator = Translator(
    model_name_or_card="seamlessM4T_large",
    vocoder_name_or_card="vocoder_36langs",
    device=device,
    dtype=torch.float16 if "cuda" in device.type else torch.float32,
)


def predict(
    task_name: str,
    audio_source: str,
    input_audio_mic: str | None,
    input_audio_file: str | None,
    input_text: str | None,
    source_language: str | None,
    target_language: str,
    stream: np.ndarray | None
) -> tuple[np.ndarray, tuple[int, np.ndarray] | None, str]:
    task_name = task_name.split()[0]
    source_language_code = (
        LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
    )
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

    if task_name in ["S2ST", "S2TT", "ASR"]:
        if audio_source == "microphone":
            input_data = input_audio_mic
        else:
            input_data = input_audio_file

        arr, org_sr = torchaudio.load(input_data)
        new_arr = torchaudio.functional.resample(
            arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE
        )

        # Append the arr to the stream to maintain the state
        if stream is None:
            stream = new_arr
        else:
            stream = np.concatenate([stream, new_arr])
        new_arr = stream


        max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
        if new_arr.shape[1] > max_length:
            new_arr = new_arr[:, :max_length]
            gr.Warning(
                f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used."
            )
        torchaudio.save(input_data, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
    else:
        input_data = input_text
    text_out, wav, sr = translator.predict(
        input=input_data,
        task_str=task_name,
        tgt_lang=target_language_code,
        src_lang=source_language_code,
        ngram_filtering=True,
    )
    if task_name in ["S2ST", "T2ST"]:
        return stream, (sr, wav.cpu().detach().numpy()), text_out
    else:
        return stream, None, text_out


def process_s2st_example(
    stream: np.ndarray, input_audio_file: str, target_language: str
) -> tuple[np.ndarray, tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="S2ST",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
        stream=stream
    )



demo = gr.Interface(
    process_s2st_example,
    ["state", gr.Audio(source="microphone", streaming=True)],
    ["state", "text"],
    live=True,
)

demo.launch(share=True)