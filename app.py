import gradio as gr
from tripoSR_inference import TripoSRModel

model = TripoSRModel()

def infer(image):
    input_path = "input.png"
    image.save(input_path)
    output_path = model.run(input_path)
    return output_path

demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="pil"),
    outputs=gr.File(label="Generated 3D Model"),
    title="TripoSR: 2D Image to 3D Mesh Generator"
)

if __name__ == "__main__":
    demo.launch()
