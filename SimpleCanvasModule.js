class SimpleCanvasModule {
  constructor(canvas_width, canvas_height, space_width, space_height) {
    this.canvas_width = canvas_width;
    this.canvas_height = canvas_height;
    this.space_width = space_width;
    this.space_height = space_height;

    this.canvas = document.createElement("canvas");
    this.canvas.width = canvas_width;
    this.canvas.height = canvas_height;
    this.context = this.canvas.getContext("2d");
    document.getElementById("elements").appendChild(this.canvas);
  }

  render(data) {
    const ctx = this.context;
    ctx.clearRect(0, 0, this.canvas_width, this.canvas_height);
    for (const a of data) {
      const x = (a.x / this.space_width) * this.canvas_width;
      const y = this.canvas_height - (a.y / this.space_height) * this.canvas_height;
      ctx.beginPath();
      ctx.arc(x, y, a.r || 2, 0, 2 * Math.PI);
      ctx.fillStyle = a.Color || "#1f77b4";
      ctx.fill();
    }
  }
}