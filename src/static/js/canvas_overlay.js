// canvas_overlay.js - HTML5 Canvas drawing layer module for head pose axes

const canvas = document.getElementById("canvas-overlay");
const videoFeed = document.getElementById("video-feed");
let ctx = null;

if (canvas) {
    ctx = canvas.getContext("2d");
}

let scaleX = 1;
let scaleY = 1;

export function setupCanvas() {
    if (!canvas || !videoFeed) return;
    
    // Synchronize canvas size and position with the video feed dimensions
    resizeCanvas();
    
    // Setup listeners
    window.addEventListener("resize", resizeCanvas);
    videoFeed.addEventListener("load", resizeCanvas);
}

export function resizeCanvas() {
    if (!canvas || !videoFeed || videoFeed.classList.contains("hidden")) {
        clearCanvas();
        return;
    }
    
    // Position canvas exactly on top of the contain-scaled image
    canvas.style.width = `${videoFeed.clientWidth}px`;
    canvas.style.height = `${videoFeed.clientHeight}px`;
    canvas.style.top = `${videoFeed.offsetTop}px`;
    canvas.style.left = `${videoFeed.offsetLeft}px`;
    
    canvas.width = videoFeed.clientWidth;
    canvas.height = videoFeed.clientHeight;
}

export function clearCanvas() {
    if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

export function drawOverlay(axes, isDistracted) {
    if (!ctx || !canvas || !videoFeed || videoFeed.classList.contains("hidden")) return;
    
    clearCanvas();
    if (!axes) return;
    
    // Dynamically calculate scaling factor relative to 640x480 source resolution
    // Since backend projects coordinates based on the frame dimensions, we read it
    const sourceW = 640;
    const sourceH = 480;
    
    scaleX = canvas.width / sourceW;
    scaleY = canvas.height / sourceH;
    
    const origin = [axes.origin[0] * scaleX, axes.origin[1] * scaleY];
    const x = [axes.x_axis[0] * scaleX, axes.x_axis[1] * scaleY];
    const y = [axes.y_axis[0] * scaleX, axes.y_axis[1] * scaleY];
    const z = [axes.z_axis[0] * scaleX, axes.z_axis[1] * scaleY];
    
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    
    // X Axis (Pitch - Red)
    ctx.strokeStyle = "#f43f5e"; // Rose 500
    ctx.shadowColor = "rgba(244, 63, 94, 0.4)";
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(origin[0], origin[1]);
    ctx.lineTo(x[0], x[1]);
    ctx.stroke();
    
    // Y Axis (Yaw - Green)
    ctx.strokeStyle = "#10b981"; // Emerald 500
    ctx.shadowColor = "rgba(16, 185, 129, 0.4)";
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(origin[0], origin[1]);
    ctx.lineTo(y[0], y[1]);
    ctx.stroke();
    
    // Z Axis (Roll - Blue)
    ctx.strokeStyle = "#3b82f6"; // Blue 500
    ctx.shadowColor = "rgba(59, 130, 246, 0.4)";
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(origin[0], origin[1]);
    ctx.lineTo(z[0], z[1]);
    ctx.stroke();
    
    // Origin (Nose tip) dot
    ctx.shadowBlur = 0;
    ctx.fillStyle = isDistracted ? "#f43f5e" : "#a855f7"; // purple when focused, red when distracted
    ctx.beginPath();
    ctx.arc(origin[0], origin[1], 4.5, 0, 2 * Math.PI);
    ctx.fill();
    
    // Origin border ring
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(origin[0], origin[1], 4.5, 0, 2 * Math.PI);
    ctx.stroke();
}
