import gradio as gr
import app
import os
import glob
import shutil
import os

def load_links():
    if os.path.exists("yt-link.txt"):
        with open("yt-link.txt", "r") as f:
            return f.read()
    return ""

def load_pdf():
    pdf_files = glob.glob(os.path.join("merged","**", "*.pdf"))
    pdf_files.sort(key=os.path.getmtime, reverse=True)  # file mới nhất lên đầu
    return pdf_files

def save_uploaded_videos(files):
    save_dir = "cache/video"
    os.makedirs(save_dir, exist_ok=True)

    # 👇 FIX: đảm bảo luôn là list
    if isinstance(files, str):
        files = [files]

    for file_path in files:
        if not file_path:
            continue

        filename = os.path.basename(file_path)
        save_path = os.path.join(save_dir, filename)

        shutil.copy(file_path, save_path)


def toggle_source(source):
    if source == "YouTube Links":
        return (
            gr.update(visible=True),   # links_input
            gr.update(visible=False)
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=True)
        )

def run_tab_scraper(source_type, link_input, videos, interval_sec, start_sec, end_sec, cleanup):
    try:
        if source_type == "YouTube Links":
            type = "links"
        else:
            type = "local"

        app.run_process(
            float(interval_sec),
            float(start_sec) if start_sec else 0,
            float(end_sec) if end_sec else None,
            cleanup,
            type=type,
            links = link_input,
            videos=videos
        )
        pdf_files = load_pdf()

        if pdf_files:
            return "Process completed successfully!", pdf_files
        else:
            return "Process completed, but no PDFs found.", []

    except Exception as e:
        return f"Error: {str(e)}", []

css = """
.gradio-container {
    max-width: 900px;
    margin: auto;
}
"""
# ==============================================================================    

# Create Gradio interface
with gr.Blocks(title="Tab Scraper") as demo:
    gr.Markdown("# Tab Scraper GUI")
    gr.Markdown("If you want to scrape from YouTube, select 'YouTube Links' and enter the links. If you want to process local videos, select 'Local Videos' and the most recent videos in the cache will be used.")

    with gr.Column():        
        source_type = gr.Radio(
            choices=["YouTube Links", "Local Videos"],
            value="YouTube Links",
            label="Select Source"
        )
        
        links_input = gr.Textbox(
            label="YouTube Links (one per line)",
            lines=5,
            value=load_links(),
            placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=..."
        )

        #Upload video từ PC
        upload_video = gr.File(label="Upload Video", file_types=[".mp4", ".mkv", ".avi", ".mov"], interactive=True)
        upload_video.upload(
            fn=save_uploaded_videos,
            inputs=upload_video
        )

        source_type.change(
            fn=toggle_source,
            inputs=source_type,
            outputs=[links_input, upload_video]
        )   

    with gr.Row():
        interval_input = gr.Number(label="Capture Interval (seconds)", value=15, minimum=1)
        start_input = gr.Number(label="Start Time (seconds)", value=0, minimum=0)
        end_input = gr.Number(label="End Time (seconds, leave empty for full)", value=None)

    cleanup_checkbox = gr.Checkbox(label="Cleanup temporary files after processing", value=True)

    run_button = gr.Button("Run Process")

    output_text = gr.Textbox(label="Output", lines=5, interactive=False)
    output_files = gr.Files(label="Generated PDFs")

    run_button.click(
        fn=run_tab_scraper,
        inputs=[
            source_type,
            links_input,
            upload_video,
            interval_input,
            start_input,
            end_input,
            cleanup_checkbox
        ],
        outputs=[output_text, output_files]
    )

    demo.load(
        fn=load_pdf,
        inputs=None,
        outputs=output_files
    )

    demo.load(
        fn=toggle_source,
        inputs=source_type,
        outputs=[links_input, upload_video]
    )

if __name__ == "__main__":
    demo.launch(
        css=css
    )
    