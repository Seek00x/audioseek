"""Local recording search UI."""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import socket
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
from .io import CONFIG_ROOT, DEFAULT_ROOT

CSS = """
.gradio-container { max-width: 1280px !important; margin: auto !important; }
#recording-header { padding: 14px 0 6px; }
#recording-header h1 { font-size: 36px; margin: 0; letter-spacing: -1px; }
#recording-header p { margin: 6px 0 12px; color: #596a78; font-size: 16px; }
.recording-card { padding: 14px 16px; }
.recording-card .rank { color: #087c70; font-weight: 650; margin-bottom: 8px; }
.recording-card .passage { white-space: pre-wrap; line-height: 1.55; font-size: 16px; }
.recording-card .metadata { color: #596a78; font-size: 13px; margin-top: 10px; }
.evidence-panel:not(:has(.recording-card)) { display: none; }
"""
CARD_COUNT = 5
# Gradio chooses built-in control text from navigator when the page mounts.
# Set the language of this recording page before mounting, without changing the
# user's saved browser settings.
ENGLISH_HEAD = """<script>
Object.defineProperty(window.navigator, 'language', {value: 'en-US', configurable: true});
Object.defineProperty(window.navigator, 'languages', {value: ['en-US', 'en'], configurable: true});
document.documentElement.lang = 'en';
</script>"""


class EnglishRecordingPage:
    """Insert locale bootstrap before Gradio mounts its built-in controls."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "GET" or scope.get("path") != "/":
            return await self.app(scope, receive, send)
        start = None
        chunks = []

        async def intercept(message):
            nonlocal start
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                if b"text/html" in headers.get(b"content-type", b"") and not headers.get(b"content-encoding"):
                    start = message
                    return
            elif message["type"] == "http.response.body" and start is not None:
                chunks.append(message.get("body", b""))
                if message.get("more_body", False):
                    return
                body = b"".join(chunks).replace(b"<head>", b"<head>" + ENGLISH_HEAD.encode("utf-8"), 1)
                headers = [(k, v) for k, v in start["headers"] if k.lower() not in [b"content-length", b"etag"]]
                headers.append((b"content-length", str(len(body)).encode("ascii")))
                await send({**start, "headers": headers})
                await send({"type": "http.response.body", "body": body, "more_body": False})
                return
            await send(message)

        await self.app(scope, receive, intercept)
PLAY_EXCERPT_JS = """async () => {
    for (let attempt = 0; attempt < 40; attempt++) {
        const root = document.getElementById('recording-player');
        const button = root?.querySelector('button.play-pause-button');
        if (button && !button.disabled) {
            const label = button.getAttribute('aria-label');
            if (label === 'Play' || label === '播放') button.click();
            root.scrollIntoView({block: 'nearest', behavior: 'smooth'});
            return [];
        }
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    return [];
}"""
def timestamp(seconds):
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{remaining:04.1f}" if hours else f"{minutes:02d}:{remaining:04.1f}"


def english_error(error):
    return getattr(error, "display_message", None) or (str(error) if isinstance(error, ValueError) else "The operation could not complete. Retry or choose another recording; details are in the app terminal.")


def create_app(service=None, root=DEFAULT_ROOT):
    import gradio as gr
    from .controls import InteractiveService, UploadStopped, lexical_notice
    service = service or InteractiveService(root)
    service.load_asr()

    def choices(state):
        return service.choices(state)

    def reset(state, message="Enter a question to find candidate passages."):
        state = {**state, "results": [], "selected_source": None}
        values = [state, message, gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                  "Choose a result to prepare its audio excerpt.", gr.update(value=None, autoplay=False)]
        for _ in range(CARD_COUNT):
            values.extend(["", gr.update(interactive=False)])
        return values

    def player(index, source, state, autoplay=False):
        if state.get("selected_source") != source or index is None:
            return gr.update(choices=[], value=None), "Search the current recording first.", gr.update(value=None, autoplay=False)
        result = state["results"][int(index)]
        ranges = service.ranges_for_result(result, source, state)
        if not ranges:
            reason = "This result has no complete usable timestamps; text evidence remains available."
            return gr.update(choices=[], value=None), reason, gr.update(value=None, autoplay=False)
        # A disconnected first interval can be only a few words. Preview the longest
        # recorded interval; this does not change rankings or invent missing times.
        first = max(range(len(ranges)), key=lambda i: ranges[i][1] - ranges[i][0])
        clip, bounds = service.result_clip(result, first, source, state)
        options = [(f"Excerpt {i+1} · {timestamp(a)}–{timestamp(b)}", str(i)) for i, (a, b) in enumerate(ranges)]
        caption = (f"**Time in uploaded recording: {timestamp(bounds[0])}–{timestamp(bounds[1])}**\n\n"
                   "ASR-estimated timestamps. The player's 00:00 is the start of this excerpt.")
        cached_clip = audio.move_resource_to_block_cache(str(clip))
        return gr.update(choices=options, value=str(first)), caption, gr.update(value=cached_clip, autoplay=autoplay)

    def select(index, source, state, autoplay=False):
        return [gr.update(value=str(index)), *player(index, source, state, autoplay)]

    def card(result, method, source, state, index):
        ranges = service.ranges_for_result(result, source, state)
        timing = " · ".join(f"{timestamp(a)}–{timestamp(b)}" for a, b in ranges) if ranges else "Audio unavailable"
        speakers = [name for name in result["speaker_ids"] if name != "Unknown"]
        speaker = "Speakers: " + ", ".join(speakers) if speakers else "Speaker identification unavailable"
        return (f'<article class="recording-card"><div class="rank">Result {index+1} · {html.escape(method)}</div>'
                f'<div class="passage">{html.escape(result["text"])}</div>'
                f'<div class="metadata">{html.escape(timing)}<br>{html.escape(speaker)}</div></article>')

    def search(source, question, method, state):
        try:
            results = service.search(source, question.strip(), method, state)
            if not results:
                return reset(state, "No candidate passages are available for this recording.")
            state = {**state, "results": results, "selected_source": source}
            items = [(f"Result {i+1} · {r['text'].replace(chr(10), ' ')[:65]}…", str(i)) for i, r in enumerate(results)]
            notice = lexical_notice(question, results)
            feedback_text = f"{len(results)} candidate passage(s) · {method}"
            if notice:
                feedback_text += "\n\n**" + notice + "**"
            values = [state, feedback_text, gr.update(choices=items, value="0"),
                      *player("0", source, state)]
            for i in range(CARD_COUNT):
                if i < len(results):
                    playable = bool(service.ranges_for_result(results[i], source, state))
                    values.extend([card(results[i], method, source, state, i),
                        gr.update(value=f"Listen to result {i+1}" if playable else "Text only", interactive=playable)])
                else:
                    values.extend(["", gr.update(interactive=False)])
            return values
        except Exception as error:
            logging.exception("Recording UI search failed")
            return reset(state, english_error(error))

    def upload_audio(path, state, progress=gr.Progress()):
        disabled = [gr.update(interactive=False) for _ in range(7)]
        disabled[4] = gr.update(value="", interactive=False)
        enabled = [gr.update(interactive=True) for _ in range(7)]
        yield [gr.update(interactive=False), "Processing English audio...", None, *reset(state), *disabled,
               gr.update(interactive=True), "Stop processing saves completed blocks after the current block finishes."]
        try:
            result = service.process_upload(path, state, lambda value, message: progress(value, desc=message))
            state = {**state, "upload_key": result["key"]}
            duration = result["audio"]["duration_seconds"] / 60
            timing = "Verified session cache reused" if result["cache_reused"] else f"Processed in {result['timings']['total_seconds']:.1f} seconds"
            recovered = result["asr_info"].get("completed_blocks_reused", 0)
            if recovered:
                timing += f"; restored {recovered} completed transcription block(s)"
            ready = f"**Ready · {duration:.1f} minutes · {result['chunk_count']} passages**\n\n{timing}. English automatic transcription; estimated timestamps."
            downloads = [transcript_downloads.move_resource_to_block_cache(str(Path(result["directory"]) / name))
                         for name in ["transcript.txt", "transcript.json"]]
            yield [gr.update(choices=choices(state), value="uploaded", interactive=True), ready, downloads,
                   *reset(state, "Recording ready. Enter your English question."), *enabled, gr.update(interactive=False), ""]
        except UploadStopped as error:
            yield [gr.update(interactive=True), english_error(error), None, *reset(state), *enabled,
                   gr.update(interactive=False), "Stopped. Retry processing continues with saved blocks in this session."]
        except Exception as error:
            logging.exception("Recording UI upload failed")
            yield [gr.update(interactive=True), "Upload failed. " + english_error(error), None, *reset(state), *enabled,
                   gr.update(interactive=False), "Fix the problem, then use Retry processing for this file."]

    def request_stop(state):
        if service.request_stop(state):
            return "Stop requested. Waiting for the current transcription block to finish and save; this can take a while.", gr.update(interactive=False)
        return "No processing task is currently running.", gr.update(interactive=False)

    def change_source(state):
        return [gr.update(value=""), *reset(state)]

    def change_interval(index, result_index, source, state):
        if index is None or result_index is None or state.get("selected_source") != source:
            return "Choose a result from the current search.", gr.update(value=None, autoplay=False)
        try:
            clip, bounds = service.result_clip(state["results"][int(result_index)], int(index), source, state)
            return (f"**Time in uploaded recording: {timestamp(bounds[0])}–{timestamp(bounds[1])}**\n\nASR-estimated timestamps; player time is relative to the excerpt.",
                    gr.update(value=audio.move_resource_to_block_cache(str(clip)), autoplay=True))
        except ValueError as error:
            return english_error(error), gr.update(value=None, autoplay=False)

    def clear(state):
        state = service.clear_upload(state)
        return [gr.update(choices=choices(state), value=None), None,
                "Upload cleared. This session's working audio, transcript, index and excerpts were removed.",
                None, *reset(state), gr.update(value=""), gr.update(interactive=False), ""]

    with gr.Blocks(title="AudioSeek · Find and replay evidence", analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        state = gr.State(None, time_to_live=86400,
                         delete_callback=lambda value: service.clear_upload(value) if isinstance(value, dict) else None)
        gr.HTML('<div id="recording-header"><h1>AudioSeek<span style="color:#087c70">.</span></h1><p>Remember the topic. Find the evidence. Listen again.</p></div>')
        with gr.Row():
            with gr.Column(scale=1, min_width=330):
                gr.Markdown("### 1 · Choose a recording")
                source = gr.Dropdown([], value=None, label="Recording", interactive=False)
                upload = gr.File(label="Upload English audio", file_types=service.config["allowed_extensions"], type="filepath", height=120)
                gr.Markdown(f"WAV / MP3 / M4A / FLAC / OGG · Up to {service.config['max_duration_seconds']/60:g} minutes and {service.config['max_file_bytes']/1024**2:g} MiB.\n\nAudio is processed on the machine running this app.")
                status = gr.Markdown("Upload a recording to transcribe and search.")
                with gr.Row():
                    replace = gr.Button("Replace / upload again", size="sm")
                    clear_button = gr.Button("Clear upload", size="sm")
                with gr.Row():
                    retry = gr.Button("Retry processing", size="sm", interactive=False)
                    stop = gr.Button("Stop processing", size="sm", interactive=False)
                stop_status = gr.Markdown("")
                transcript_downloads = gr.File(label="Download transcript", interactive=False, file_count="multiple")
                gr.Markdown("### 2 · Ask a question")
                query = gr.Textbox(label="English question", value="", lines=2, placeholder="Ask a question about the selected recording.")
                method = gr.Radio(["BM25", "SBERT", "Hybrid"], value="Hybrid", label="Retrieval method")
                submit = gr.Button("Find passages", variant="primary")
            with gr.Column(scale=2, min_width=480):
                gr.Markdown("### 3 · Read and listen")
                gr.Markdown("Candidate passages — verify by reading and listening. Ranked results can be unrelated when the recording does not answer the question.")
                feedback = gr.Markdown("Enter a question to find candidate passages.")
                with gr.Row():
                    selected = gr.Dropdown([], label="Selected result", interactive=True)
                    interval = gr.Dropdown([], label="Playback excerpt", interactive=True)
                caption = gr.Markdown("Choose a result to prepare its audio excerpt.")
                audio = gr.Audio(label="Excerpt player", type="filepath", format="wav", interactive=False, autoplay=False,
                                 buttons=["download"], elem_id="recording-player")
                card_components = []
                card_buttons = []
                for i in range(CARD_COUNT):
                    with gr.Column(elem_classes=["evidence-panel"]):
                        passage = gr.HTML("")
                        listen = gr.Button(f"Listen to result {i+1}", size="sm", interactive=False)
                    card_components.extend([passage, listen])
                    card_buttons.append(listen)
        view_outputs = [state, feedback, selected, interval, caption, audio, *card_components]
        player_outputs = [selected, interval, caption, audio]
        controls = [upload, replace, clear_button, retry, query, method, submit]
        demo.load(service.new_session, outputs=state, api_name="initialize_session")
        source.input(change_source, state, [query, *view_outputs])
        method.input(reset, state, view_outputs)
        query.input(reset, state, view_outputs, show_progress="hidden")
        submit.click(search, [source, query, method, state], view_outputs, api_name="search", show_progress_on=[feedback])
        query.submit(search, [source, query, method, state], view_outputs, show_progress_on=[feedback])
        selected.input(lambda index, src, st: player(index, src, st), [selected, source, state], [interval, caption, audio], api_name="select_result")
        interval.input(change_interval, [interval, selected, source, state], [caption, audio], api_name="select_interval")
        for i, button in enumerate(card_buttons):
            def listen_to(src, st, index=i):
                return select(str(index), src, st, autoplay=True)
            button.click(listen_to, [source, state], player_outputs, api_name=f"listen_result_{i+1}", show_progress="hidden").then(
                fn=None, inputs=None, outputs=None, js=PLAY_EXCERPT_JS)
        replace.click(lambda: None, outputs=upload, api_name="replace_upload")
        upload_outputs = [source, status, transcript_downloads, *view_outputs, *controls, stop, stop_status]
        upload.upload(upload_audio, [upload, state], upload_outputs, api_name="upload_recording", show_progress_on=[status])
        retry.click(upload_audio, [upload, state], upload_outputs, api_name="retry_processing", show_progress_on=[status])
        stop.click(request_stop, state, [stop_status, stop], queue=False, concurrency_limit=None, api_name="stop_processing")
        clear_button.click(clear, state, [source, upload, status, transcript_downloads, *view_outputs, query, retry, stop_status], api_name="clear_upload")
        upload.clear(clear, state, [source, upload, status, transcript_downloads, *view_outputs, query, retry, stop_status])
    demo.queue(default_concurrency_limit=1, max_size=12)
    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch AudioSeek: local recording search and playback.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--download-models", action="store_true", help="Download and verify model files, then exit")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT, help="Model cache and session working directory")
    args = parser.parse_args()
    from .downloads import ModelDownloadError, download_models
    if args.download_models:
        try:
            download_models(args.data_dir, report=lambda message: print(message, flush=True))
        except ModelDownloadError as error:
            parser.exit(1, f"Model setup failed: {error}\n")
        return
    if not 1024 <= args.port <= 65535:
        raise ValueError("Choose a port between 1024 and 65535.")
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as error:
            from urllib.request import ProxyHandler, build_opener
            address = f"http://127.0.0.1:{args.port}"
            try:
                with build_opener(ProxyHandler({})).open(address + "/config", timeout=2) as response:
                    running = json.load(response)
                is_recording_app = running.get("title") == "AudioSeek · Find and replay evidence"
            except Exception:
                is_recording_app = False
            if is_recording_app:
                print(f"AudioSeek is already running: {address}", flush=True)
                if args.open_browser:
                    import webbrowser
                    webbrowser.open(address)
                return
            raise RuntimeError(f"Port {args.port} is already in use. Choose another --port; the existing process was not stopped.") from error
    print("AudioSeek is starting. First launch downloads approximately 240 MB of model files.", flush=True)
    print("Watch the four setup stages below. The browser page becomes available after model loading.", flush=True)
    print("If a download fails, rerun this command; completed, verified files will be reused.", flush=True)
    import gradio as gr
    from starlette.middleware import Middleware
    try:
        demo = create_app(root=args.data_dir)
    except ModelDownloadError as error:
        parser.exit(1, f"Model setup failed: {error}\n")
    print("Models loaded. Starting the local page...", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=args.open_browser,
        max_file_size=json.loads((CONFIG_ROOT / "upload.json").read_text("utf-8"))["max_file_bytes"], theme=gr.themes.Soft(primary_hue="teal", font=["Arial", "sans-serif"]),
        css=CSS, app_kwargs={"middleware": [Middleware(EnglishRecordingPage)]}, footer_links=[], show_error=False)


if __name__ == "__main__":
    main()
