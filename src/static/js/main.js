// main.js - Central orchestrator module for the F1 Telemetry dashboard

import { initCharts, updateCharts, resetCharts } from './charts.js?v=2';
import { updateTelemetryCounters, updateEngagementGauge, appendLog, generateSummary, clearLogs } from './telemetry.js?v=2';
import { setupCanvas, resizeCanvas, drawOverlay, clearCanvas } from './canvas_overlay.js?v=2';

// Application State
let sessionId = null;
let currentSource = "camera";
let uploadedFilename = null;
let isSessionActive = false;
let socket = null;

// Telemetry History (Keep last 300 data points for 60 seconds at 5 Hz)
const MAX_CHART_POINTS = 60;
let earHistory = [];
let marHistory = [];
let poseHistory = { pitch: [], yaw: [], roll: [] };
let chartTimestamps = [];
let frameCount = 0;
let lastChartUpdateTime = 0;

// DOM Elements
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnReset = document.getElementById("btn-reset");
const btnSummary = document.getElementById("btn-summary");
const btnClearLogs = document.getElementById("btn-clear-logs");

const srcWebcam = document.getElementById("src-webcam");
const srcFile = document.getElementById("src-file");
const webcamSettings = document.getElementById("webcam-settings");
const fileSettings = document.getElementById("file-settings");
const fileInput = document.getElementById("video-file-input");
const btnUploadFile = document.getElementById("btn-upload-file");
const selectedFileLabel = document.getElementById("selected-file-label");

const thresholdEar = document.getElementById("threshold-ear");
const thresholdMar = document.getElementById("threshold-mar");
const performanceSkip = document.getElementById("performance-skip");
const performanceScale = document.getElementById("performance-scale");

const valEar = document.getElementById("ear-val");
const valMar = document.getElementById("mar-val");
const valSkip = document.getElementById("skip-val");
const valScale = document.getElementById("scale-val");

const videoFeed = document.getElementById("video-feed");
const videoPlaceholder = document.getElementById("video-placeholder");
const liveIndicator = document.getElementById("live-indicator");
const streamStatus = document.getElementById("stream-status");
const sessionIdDisplay = document.getElementById("session-id-display");
const summaryModal = document.getElementById("summary-modal");
const btnCloseModal = document.getElementById("btn-close-modal");
const btnCopyPrompt = document.getElementById("btn-copy-prompt");
const sumCoachingPrompt = document.getElementById("sum-coaching-prompt");

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    setupCanvas();
    setupEventListeners();
    connectWebSocket();
    setupPortal();
});

function setupEventListeners() {
    // Slider Syncs
    if (thresholdEar) thresholdEar.addEventListener("input", (e) => valEar.textContent = parseFloat(e.target.value).toFixed(2));
    if (thresholdMar) thresholdMar.addEventListener("input", (e) => valMar.textContent = parseFloat(e.target.value).toFixed(2));
    if (performanceSkip) performanceSkip.addEventListener("input", (e) => valSkip.textContent = e.target.value);
    if (performanceScale) performanceScale.addEventListener("input", (e) => valScale.textContent = parseFloat(e.target.value).toFixed(2));

    // Video Source Toggles
    if (srcWebcam) {
        srcWebcam.addEventListener("click", () => {
            currentSource = "camera";
            srcWebcam.className = "py-2 px-3 rounded-lg text-xs font-semibold transition bg-cyan-600 text-white shadow shadow-cyan-950/20";
            srcFile.className = "py-2 px-3 rounded-lg text-xs font-semibold transition text-slate-400 hover:text-slate-200";
            if (webcamSettings) webcamSettings.classList.remove("hidden");
            if (fileSettings) fileSettings.classList.add("hidden");
        });
    }

    if (srcFile) {
        srcFile.addEventListener("click", () => {
            currentSource = "video_file";
            srcFile.className = "py-2 px-3 rounded-lg text-xs font-semibold transition bg-cyan-600 text-white shadow shadow-cyan-950/20";
            srcWebcam.className = "py-2 px-3 rounded-lg text-xs font-semibold transition text-slate-400 hover:text-slate-200";
            if (fileSettings) fileSettings.classList.remove("hidden");
            if (webcamSettings) webcamSettings.classList.add("hidden");
        });
    }

    // File Upload Trigger
    if (btnUploadFile) btnUploadFile.addEventListener("click", () => fileInput.click());
    if (fileInput) fileInput.addEventListener("change", handleFileUpload);

    // Session Operations
    if (btnStart) btnStart.addEventListener("click", startSession);
    if (btnStop) btnStop.addEventListener("click", stopSession);
    if (btnReset) btnReset.addEventListener("click", resetSession);
    if (btnSummary) btnSummary.addEventListener("click", () => generateSummary(sessionId));

    // Clear Logs
    if (btnClearLogs) {
        btnClearLogs.addEventListener("click", clearLogs);
    }

    // Modal Close
    if (btnCloseModal) btnCloseModal.addEventListener("click", () => summaryModal.classList.add("hidden"));
    if (btnCopyPrompt) {
        btnCopyPrompt.addEventListener("click", () => {
            navigator.clipboard.writeText(sumCoachingPrompt.value);
            const originalText = btnCopyPrompt.textContent;
            btnCopyPrompt.textContent = "Copied!";
            btnCopyPrompt.classList.add("bg-emerald-600", "text-white");
            setTimeout(() => {
                btnCopyPrompt.textContent = originalText;
                btnCopyPrompt.classList.remove("bg-emerald-600", "text-white");
            }, 1500);
        });
    }
}

// Handle File Selection and Upload
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (selectedFileLabel) {
        selectedFileLabel.textContent = "Uploading " + file.name + "...";
        selectedFileLabel.classList.remove("text-emerald-400");
        selectedFileLabel.classList.add("text-slate-400");
    }
    appendLog("INFO", "Client", `Uploading file: ${file.name}`);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/api/session/upload", {
            method: "POST",
            body: formData
        });
        if (response.ok) {
            const data = await response.json();
            uploadedFilename = data.filename;
            if (selectedFileLabel) {
                selectedFileLabel.textContent = file.name + " (Uploaded)";
                selectedFileLabel.classList.remove("text-slate-400");
                selectedFileLabel.classList.add("text-emerald-400");
            }
            appendLog("INFO", "Client", `File upload completed: ${uploadedFilename}`);
        } else {
            throw new Error("Upload failed.");
        }
    } catch (err) {
        appendLog("ERROR", "Client", `Upload error: ${err.message}`);
        if (selectedFileLabel) {
            selectedFileLabel.textContent = "Upload failed!";
            selectedFileLabel.classList.remove("text-slate-400");
            selectedFileLabel.classList.add("text-red-400");
        }
    }
}

// Start Session API call
async function startSession() {
    if (currentSource === "video_file" && !uploadedFilename) {
        alert("Please select and upload a video file first.");
        return;
    }

    appendLog("INFO", "Client", "Initiating session start request...");
    
    const config = {
        video_source: currentSource,
        camera_id: document.getElementById("camera-id").value,
        video_filename: uploadedFilename,
        ear_threshold: parseFloat(thresholdEar.value),
        mar_threshold: parseFloat(thresholdMar.value),
        frame_skip: parseInt(performanceSkip.value),
        resize_scale: parseFloat(performanceScale.value)
    };

    try {
        const response = await fetch("/api/session/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            // Reset metrics histories for the new session
            earHistory = [];
            marHistory = [];
            poseHistory = { pitch: [], yaw: [], roll: [] };
            chartTimestamps = [];
            lastChartUpdateTime = 0;
            resetCharts();

            const data = await response.json();
            sessionId = data.session_id;
            if (sessionIdDisplay) sessionIdDisplay.textContent = sessionId;
            isSessionActive = true;

            // UI State Change
            if (btnStart) {
                btnStart.disabled = true;
                btnStart.classList.add("opacity-50", "pointer-events-none");
            }
            if (btnStop) {
                btnStop.disabled = false;
                btnStop.classList.remove("opacity-50", "bg-red-600/30", "text-slate-400", "pointer-events-none");
                btnStop.classList.add("bg-red-600", "text-white");
            }
            if (btnSummary) {
                btnSummary.disabled = false;
                btnSummary.classList.remove("opacity-50", "pointer-events-none");
            }

            // Activate Video stream
            if (videoFeed) {
                videoFeed.src = "/api/video-feed?t=" + new Date().getTime(); // force reload
                videoFeed.classList.remove("hidden");
            }
            if (videoPlaceholder) videoPlaceholder.classList.add("hidden");
            if (liveIndicator) liveIndicator.classList.remove("hidden");

            if (streamStatus) {
                streamStatus.innerHTML = '<span class="w-2.5 h-2.5 mr-1.5 rounded-full bg-emerald-500 glow-green animate-pulse"></span>Active';
                streamStatus.className = "inline-flex items-center text-xs font-semibold text-emerald-400";
            }
            
            // Sync canvas dimensions
            resizeCanvas();
            
            appendLog("INFO", "Client", `Session ${sessionId} running successfully.`);
        } else {
            const error = await response.json();
            throw new Error(error.detail || "Server error.");
        }
    } catch (err) {
        appendLog("ERROR", "Client", `Failed to start session: ${err.message}`);
    }
}

// Stop Session API call
async function stopSession() {
    appendLog("INFO", "Client", "Requesting session stop...");
    try {
        const response = await fetch("/api/session/stop", { method: "POST" });
        if (response.ok) {
            handleSessionEnd();
        }
    } catch (err) {
        appendLog("ERROR", "Client", `Stop request failed: ${err.message}`);
    }
}

// Local Session End UI cleanup
function handleSessionEnd() {
    isSessionActive = false;
    
    // UI resets
    if (btnStart) {
        btnStart.disabled = false;
        btnStart.classList.remove("opacity-50", "pointer-events-none");
    }
    if (btnStop) {
        btnStop.disabled = true;
        btnStop.classList.add("opacity-50", "bg-red-600/30", "text-slate-400", "pointer-events-none");
        btnStop.classList.remove("bg-red-600", "text-white");
    }

    // Close stream feed in UI
    if (videoFeed) {
        videoFeed.src = "";
        videoFeed.classList.add("hidden");
    }
    if (videoPlaceholder) videoPlaceholder.classList.remove("hidden");
    if (liveIndicator) liveIndicator.classList.add("hidden");

    if (streamStatus) {
        streamStatus.innerHTML = '<span class="w-2 h-2 mr-1.5 rounded-full bg-slate-500"></span>Idle';
        streamStatus.className = "inline-flex items-center text-xs font-semibold text-slate-400";
    }
    
    clearCanvas();
    appendLog("INFO", "Client", "Session analyzer stopped.");
}

// Reset Session logic
async function resetSession() {
    appendLog("INFO", "Client", "Resetting session...");
    try {
        await fetch("/api/session/reset", { method: "POST" });
        
        // Reset metrics histories
        earHistory = [];
        marHistory = [];
        poseHistory = { pitch: [], yaw: [], roll: [] };
        chartTimestamps = [];
        lastChartUpdateTime = 0;
        
        // Redraw blank charts
        resetCharts();
        clearCanvas();

        // Reset Counters
        const statBlinks = document.getElementById("stat-blinks");
        const statYawns = document.getElementById("stat-yawns");
        const statLookingAway = document.getElementById("stat-looking-away-val");
        const statFps = document.getElementById("stat-fps");
        const statPose = document.getElementById("stat-pose");

        if (statBlinks) statBlinks.textContent = "0";
        if (statYawns) statYawns.textContent = "0";
        if (statLookingAway) statLookingAway.textContent = "0";
        if (statFps) statFps.textContent = "FPS: --";
        if (statPose) statPose.textContent = "P:-- Y:-- R:--";
        updateEngagementGauge(0);
        
        sessionId = null;
        if (sessionIdDisplay) sessionIdDisplay.textContent = "--------";
        if (btnSummary) {
            btnSummary.disabled = true;
            btnSummary.classList.add("opacity-50", "pointer-events-none");
        }
        
        handleSessionEnd();
        appendLog("INFO", "Client", "Session metrics reset completed.");
    } catch (err) {
        appendLog("ERROR", "Client", `Reset failed: ${err.message}`);
    }
}

// Connect WebSockets
function connectWebSocket() {
    if (window.location.protocol === "file:") {
        const errorMsg = "Dashboard loaded as a local file (file://). Telemetry requires the backend server. Please run 'python main.py dashboard' and open http://localhost:8001 in your browser.";
        console.error(errorMsg);
        alert(errorMsg);
        appendLog("ERROR", "Client", errorMsg);
        return;
    }
    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProto}//${window.location.host}/api/ws/telemetry`;
    const wsStatus = document.getElementById("ws-status");
    
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        if (wsStatus) {
            wsStatus.innerHTML = '<span class="w-2.5 h-2.5 mr-1.5 rounded-full bg-emerald-500 glow-green animate-pulse"></span>Connected';
            wsStatus.className = "inline-flex items-center text-xs font-semibold text-emerald-400";
        }
        const diagWs = document.getElementById("diag-ws");
        if (diagWs) {
            diagWs.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-550"></span>ONLINE';
            diagWs.className = "text-emerald-400 font-bold flex items-center gap-1";
        }
        appendLog("INFO", "WebSocket", "Telemetry stream connection established.");
    };

    socket.onclose = () => {
        if (wsStatus) {
            wsStatus.innerHTML = '<span class="w-2 h-2 mr-1.5 rounded-full bg-red-500 glow-red animate-pulse"></span>Reconnecting...';
            wsStatus.className = "inline-flex items-center text-xs font-semibold text-amber-400";
        }
        const diagWs = document.getElementById("diag-ws");
        if (diagWs) {
            diagWs.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse"></span>RECONNECTING';
            diagWs.className = "text-amber-400 font-bold flex items-center gap-1";
        }
        appendLog("WARNING", "WebSocket", "Connection lost. Reconnecting in 3 seconds...");
        // NOTE: Do NOT call handleSessionEnd() here — the backend session may still
        // be running. Killing isSessionActive causes all telemetry to be silently
        // dropped after reconnect, which was the root cause of the "no metrics" bug.
        setTimeout(connectWebSocket, 3000);
    };

    socket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            if (message.type === "log") {
                appendLog(message.data.level, message.data.name, message.data.message, message.data.traceback);
            } else if (message.type === "telemetry") {
                if (!isSessionActive) {
                    isSessionActive = true;
                    console.log("Session re-activated from incoming telemetry");
                }
                handleTelemetryData(message.data);
            } else if (message.type === "finished") {
                console.log("Finished message received");
                appendLog("INFO", "Stream", "Source stream processed to end.");
                handleSessionEnd();
                generateSummary(sessionId); // Auto-popup summary when file processing ends
            }
        } catch (err) {
            console.error("Error parsing WS message:", err);
        }
    };
}

// Process received WebSocket telemetry
function handleTelemetryData(telemetry) {
    // Update numeric stats counters
    updateTelemetryCounters(telemetry);
    
    // Update Engagement Gauge
    updateEngagementGauge(telemetry.engagement_score);
    
    // Distracted Visualization
    const videoContainer = document.getElementById("video-container");
    const distractedBadge = document.getElementById("distracted-badge");
    if (telemetry.is_distracted) {
        if (videoContainer) {
            // Adjust glowing ring color based on distraction severity
            videoContainer.classList.remove("ring-4", "ring-rose-500", "shadow-[0_0_30px_rgba(244,63,94,0.4)]", "ring-amber-500", "shadow-[0_0_30px_rgba(245,158,11,0.4)]", "ring-slate-500", "shadow-[0_0_30px_rgba(100,116,139,0.4)]");
            if (telemetry.distraction_type === "drowsy") {
                videoContainer.classList.add("ring-4", "ring-amber-500", "shadow-[0_0_30px_rgba(245,158,11,0.4)]");
            } else if (telemetry.distraction_type === "absent") {
                videoContainer.classList.add("ring-4", "ring-slate-500", "shadow-[0_0_30px_rgba(100,116,139,0.4)]");
            } else {
                videoContainer.classList.add("ring-4", "ring-rose-500", "shadow-[0_0_30px_rgba(244,63,94,0.4)]");
            }
        }
        if (distractedBadge) {
            distractedBadge.classList.remove("hidden");
            if (telemetry.distraction_type === "drowsy") {
                distractedBadge.innerHTML = '<span class="material-symbols-outlined text-xs align-middle mr-1.5 animate-pulse">dark_mode</span>DROWSY';
                distractedBadge.className = "absolute top-4 right-4 bg-amber-950/90 text-amber-200 font-bold text-xs tracking-widest uppercase px-4 py-2 rounded-md shadow-[0_0_20px_rgba(245,158,11,0.6)] border border-amber-500/40 z-10 animate-bounce";
            } else if (telemetry.distraction_type === "absent") {
                distractedBadge.innerHTML = '<span class="material-symbols-outlined text-xs align-middle mr-1.5">person_off</span>USER ABSENT';
                distractedBadge.className = "absolute top-4 right-4 bg-slate-950/90 text-slate-300 font-bold text-xs tracking-widest uppercase px-4 py-2 rounded-md shadow-[0_0_20px_rgba(148,163,184,0.6)] border border-slate-500/40 z-10";
            } else {
                distractedBadge.innerHTML = '<span class="material-symbols-outlined text-xs align-middle mr-1.5 animate-pulse">visibility_off</span>LOOKING AWAY';
                distractedBadge.className = "absolute top-4 right-4 bg-rose-950/90 text-rose-200 font-bold text-xs tracking-widest uppercase px-4 py-2 rounded-md shadow-[0_0_20px_rgba(244,63,94,0.6)] border border-rose-500/40 z-10 animate-bounce";
            }
        }
    } else {
        if (videoContainer) videoContainer.classList.remove("ring-4", "ring-rose-500", "ring-amber-500", "ring-slate-500", "shadow-[0_0_30px_rgba(244,63,94,0.4)]", "shadow-[0_0_30px_rgba(245,158,11,0.4)]", "shadow-[0_0_30px_rgba(100,116,139,0.4)]");
        if (distractedBadge) distractedBadge.classList.add("hidden");
    }

    // Draw coordinate axes on canvas overlay
    drawOverlay(telemetry.pose_axes_2d, telemetry.is_distracted);

    // Skip appending to charts history if frame is a synthetic skip frame
    if (telemetry.frame_skipped) {
        return;
    }

    const currentTime = telemetry.timestamp;
    if (currentTime - lastChartUpdateTime >= 1.0) {
        // Update history arrays
        const elapsed = Math.round(currentTime % 1000); // relative time marker
        chartTimestamps.push(elapsed);
        earHistory.push(telemetry.avg_ear);
        marHistory.push(telemetry.mar);
        poseHistory.pitch.push(telemetry.pitch || 0);
        poseHistory.yaw.push(telemetry.yaw || 0);
        poseHistory.roll.push(telemetry.roll || 0);

        // Limit histories length
        if (chartTimestamps.length > MAX_CHART_POINTS) {
            chartTimestamps.shift();
            earHistory.shift();
            marHistory.shift();
            poseHistory.pitch.shift();
            poseHistory.yaw.shift();
            poseHistory.roll.shift();
        }

        // Refresh charts data exactly once per second
        updateCharts(earHistory, marHistory, poseHistory);
        lastChartUpdateTime = currentTime;
    }
}

function setupPortal() {
    // 1. Verify Webcam access via Python backend diagnostics (prevents browser hardware lock conflicts)
    fetch("/api/session/verify-camera")
        .then(res => res.json())
        .then(data => {
            const diagCamera = document.getElementById("diag-camera");
            if (diagCamera) {
                if (data.status === "detected") {
                    diagCamera.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>DETECTED';
                    diagCamera.className = "text-emerald-400 font-bold flex items-center gap-1";
                } else {
                    diagCamera.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse"></span>UNAVAILABLE';
                    diagCamera.className = "text-rose-400 font-bold flex items-center gap-1";
                }
            }
        })
        .catch(err => {
            const diagCamera = document.getElementById("diag-camera");
            if (diagCamera) {
                diagCamera.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse"></span>UNAVAILABLE';
                diagCamera.className = "text-rose-400 font-bold flex items-center gap-1";
            }
        });

    // 2. Interactive enter-cockpit sequence
    const btnEnterCockpit = document.getElementById("btn-enter-cockpit");
    const introPortal = document.getElementById("intro-portal");
    const initProgressContainer = document.getElementById("init-progress-container");
    const initProgressBar = document.getElementById("init-progress-bar");
    const initStatusText = document.getElementById("init-status-text");
    const initPercentage = document.getElementById("init-percentage");

    const dbHeader = document.getElementById("dashboard-header");
    const dbSidebar = document.getElementById("dashboard-sidebar");
    const dbMain = document.getElementById("dashboard-main");

    if (btnEnterCockpit) {
        btnEnterCockpit.addEventListener("click", () => {
            // Play futuristic Boot audio synthesizer chime!
            playBootChime();

            btnEnterCockpit.classList.add("hidden");
            if (initProgressContainer) initProgressContainer.classList.remove("hidden");

            let progress = 0;
            const interval = setInterval(() => {
                progress += 5;
                if (initProgressBar) initProgressBar.style.width = `${progress}%`;
                if (initPercentage) initPercentage.textContent = `${progress}%`;

                if (progress < 30) {
                    if (initStatusText) initStatusText.textContent = "LINKING DATA STREAMS...";
                } else if (progress < 70) {
                    if (initStatusText) initStatusText.textContent = "CALIBRATING CORRELATORS...";
                } else if (progress < 95) {
                    if (initStatusText) initStatusText.textContent = "SYSTEMS COMING ONLINE...";
                } else {
                    if (initStatusText) initStatusText.textContent = "READY.";
                }

                if (progress >= 100) {
                    clearInterval(interval);

                    // Smooth fade out of portal screen
                    if (introPortal) {
                        introPortal.classList.add("opacity-0", "scale-105", "pointer-events-none");
                        setTimeout(() => introPortal.remove(), 1200);
                    }

                    // Smooth fade in of main cockpit layout elements
                    if (dbHeader) dbHeader.classList.remove("opacity-0", "blur-md", "pointer-events-none", "scale-95");
                    if (dbSidebar) dbSidebar.classList.remove("opacity-0", "blur-md", "pointer-events-none");
                    if (dbMain) dbMain.classList.remove("opacity-0", "blur-md", "pointer-events-none", "scale-95");
                }
            }, 60);
        });
    }
}

function playBootChime() {
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        const ctx = new AudioContext();
        
        // Hum oscillator
        const osc1 = ctx.createOscillator();
        const gain1 = ctx.createGain();
        osc1.type = "sawtooth";
        osc1.frequency.setValueAtTime(80, ctx.currentTime);
        osc1.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 1.0);
        gain1.gain.setValueAtTime(0.15, ctx.currentTime);
        gain1.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 1.2);
        
        // Bell sound chime
        const osc2 = ctx.createOscillator();
        const gain2 = ctx.createGain();
        osc2.type = "sine";
        osc2.frequency.setValueAtTime(440, ctx.currentTime);
        osc2.frequency.setValueAtTime(880, ctx.currentTime + 0.15);
        osc2.frequency.setValueAtTime(1320, ctx.currentTime + 0.3);
        gain2.gain.setValueAtTime(0.12, ctx.currentTime);
        gain2.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.8);
        
        // Cyber tone filter
        const filter = ctx.createBiquadFilter();
        filter.type = "lowpass";
        filter.frequency.setValueAtTime(1200, ctx.currentTime);
        filter.Q.setValueAtTime(5, ctx.currentTime);
        
        osc1.connect(gain1);
        gain1.connect(filter);
        
        osc2.connect(gain2);
        gain2.connect(filter);
        
        filter.connect(ctx.destination);
        
        osc1.start();
        osc2.start();
        
        osc1.stop(ctx.currentTime + 1.2);
        osc2.stop(ctx.currentTime + 1.2);
    } catch (e) {
        console.warn("Audio Context init failed:", e);
    }
}
