// telemetry.js - Telemetry updates and terminal logging module

let logCount = 0;

const statEngagementText = document.getElementById("stat-engagement-text");
const statEngagementLabel = document.getElementById("stat-engagement-label");
const statBlinks = document.getElementById("stat-blinks");
const statYawns = document.getElementById("stat-yawns");
const statLookingAway = document.getElementById("stat-looking-away-val");
const statPose = document.getElementById("stat-pose");
const statFps = document.getElementById("stat-fps");
const gaugeCircle = document.getElementById("gauge-circle");
const statSpeaking = document.getElementById("stat-speaking");
const logConsole = document.getElementById("log-console");
const logCountDisplay = document.getElementById("log-count");

// Summary Modal Elements
const summaryModal = document.getElementById("summary-modal");
const summaryMeta = document.getElementById("summary-meta");
const sumEngagement = document.getElementById("sum-engagement");
const sumBlinkRate = document.getElementById("sum-blink-rate");
const sumBlinks = document.getElementById("sum-blinks");
const sumYawns = document.getElementById("sum-yawns");
const sumEar = document.getElementById("sum-ear");
const sumMar = document.getElementById("sum-mar");
const sumPose = document.getElementById("sum-pose");
const sumLookingAway = document.getElementById("sum-looking-away");
const sumInsightsList = document.getElementById("sum-insights-list");
const sumCoachingPrompt = document.getElementById("sum-coaching-prompt");

export function updateTelemetryCounters(telemetry) {
    if (statBlinks) statBlinks.textContent = telemetry.blink_count;
    if (statYawns) statYawns.textContent = telemetry.yawn_count;
    if (statLookingAway) statLookingAway.textContent = telemetry.looking_away_count;
    if (statFps) statFps.textContent = `FPS: ${telemetry.fps.toFixed(1)}`;
    
    if (statPose) {
        const p = telemetry.pitch || 0;
        const y = telemetry.yaw || 0;
        const r = telemetry.roll || 0;
        statPose.textContent = `P:${p.toFixed(0)}° Y:${y.toFixed(0)}° R:${r.toFixed(0)}°`;
    }

    if (statSpeaking) {
        if (telemetry.is_talking) {
            statSpeaking.classList.remove("hidden");
        } else {
            statSpeaking.classList.add("hidden");
        }
    }
}

export function updateEngagementGauge(score) {
    if (statEngagementText) {
        statEngagementText.textContent = `${score.toFixed(0)}%`;
    }
    
    if (gaugeCircle) {
        // Circle length = 2 * PI * r = 2 * 3.14159 * 28 = 175.9
        const circleLength = 175.9;
        const offset = circleLength - (score / 100) * circleLength;
        gaugeCircle.style.strokeDashoffset = offset;
    }

    if (statEngagementLabel) {
        // Label evaluation
        let label = "LOW";
        let color = "text-red-500";
        if (score >= 90) { label = "EXTREME"; color = "text-emerald-400"; }
        else if (score >= 70) { label = "HIGH"; color = "text-green-400"; }
        else if (score >= 50) { label = "ATTENTIVE"; color = "text-yellow-400"; }
        else if (score >= 30) { label = "DISTRACTED"; color = "text-amber-500"; }
        
        statEngagementLabel.textContent = label;
        statEngagementLabel.className = `text-xs font-bold mt-0.5 ${color}`;
    }
}

export function appendLog(level, name, message, traceback = null) {
    if (!logConsole || !logCountDisplay) return;
    logCount++;
    logCountDisplay.textContent = `${logCount} entries`;

    // Strip console ready text on first log
    if (logCount === 1) {
        logConsole.innerHTML = "";
    }

    let color = "text-slate-300";
    if (level === "ERROR") color = "text-red-400";
    else if (level === "WARNING") color = "text-amber-400";
    else if (level === "INFO") color = "text-cyan-400";

    const timestamp = new Date().toLocaleTimeString();
    
    const div = document.createElement("div");
    div.className = `font-mono py-0.5 text-[11px] border-b border-slate-900/20 ${color}`;
    
    let html = `<span class="text-slate-600">[${timestamp}]</span> <span class="font-bold">[${level}]</span> [${name}] — ${message}`;
    
    if (traceback) {
        const cleanTb = traceback.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        html += `<details class="ml-6 text-red-300/80 mt-1 cursor-pointer select-none text-[11px]"><summary class="outline-none">View Stack Trace</summary><pre class="bg-red-950/20 border border-red-900/20 rounded p-2 mt-1 font-mono text-[10px] overflow-x-auto whitespace-pre-wrap">${cleanTb}</pre></details>`;
    }
    
    div.innerHTML = html;
    logConsole.appendChild(div);
    
    // Auto Scroll
    logConsole.scrollTop = logConsole.scrollHeight;

    // Prune UI logs if excess to prevent memory leak (keep last 200 logs in DOM)
    if (logConsole.children.length > 200) {
        logConsole.removeChild(logConsole.firstChild);
    }
}

export function formatTime(isoStr) {
    if (!isoStr) return "—";
    try {
        const d = new Date(isoStr);
        return d.toLocaleTimeString();
    } catch {
        return isoStr;
    }
}

export function clearLogs() {
    if (logConsole && logCountDisplay) {
        logConsole.innerHTML = '<span class="text-slate-600 select-none">Terminal cleared.</span>';
        logCount = 0;
        logCountDisplay.textContent = "0 entries";
    }
}

export async function generateSummary(sessionId) {
    if (!sessionId) return;
    
    appendLog("INFO", "Client", "Fetching summary report...");
    try {
        const response = await fetch(`/api/telemetry/summary/${sessionId}`);
        if (response.ok) {
            const data = await response.json();
            
            // Populate Modal values
            if (summaryMeta) summaryMeta.textContent = `Session ID: ${data.session_id} | Started: ${formatTime(data.started_at)} | Duration: ${data.duration_seconds.toFixed(1)}s | Total Frames: ${data.total_frames}`;
            if (sumEngagement) sumEngagement.textContent = `${data.engagement_score.toFixed(1)}%`;
            if (sumBlinkRate) sumBlinkRate.textContent = `${data.blink_rate_per_min.toFixed(1)}/min`;
            if (sumBlinks) sumBlinks.textContent = data.blink_count;
            if (sumYawns) sumYawns.textContent = data.yawn_count;
            if (sumEar) sumEar.textContent = data.avg_ear ? data.avg_ear.toFixed(3) : "—";
            if (sumMar) sumMar.textContent = data.avg_mar ? data.avg_mar.toFixed(3) : "—";
            if (sumPose) sumPose.textContent = `${data.avg_pitch ? data.avg_pitch.toFixed(1) : 0}° / ${data.avg_yaw ? data.avg_yaw.toFixed(1) : 0}°`;
            if (sumLookingAway) sumLookingAway.textContent = `${data.looking_away_count}s (${(data.looking_away_ratio * 100).toFixed(1)}%)`;

            // Insights list
            if (sumInsightsList) {
                sumInsightsList.innerHTML = "";
                if (data.insights && data.insights.length > 0) {
                    const listInsights = data.insights.slice(0, -1);
                    const llmPrompt = data.insights[data.insights.length - 1];
                    
                    listInsights.forEach(ins => {
                        const li = document.createElement("li");
                        li.textContent = ins;
                        sumInsightsList.appendChild(li);
                    });
                    
                    if (sumCoachingPrompt) sumCoachingPrompt.value = llmPrompt || "";
                } else {
                    sumInsightsList.innerHTML = "<li>No specific engagement alerts or insights found.</li>";
                    if (sumCoachingPrompt) sumCoachingPrompt.value = "No prompt available.";
                }
            }

            // Show Modal
            if (summaryModal) summaryModal.classList.remove("hidden");
            appendLog("INFO", "Client", "Summary report loaded successfully.");
        } else {
            throw new Error("Report not found on backend.");
        }
    } catch (err) {
        appendLog("ERROR", "Client", `Failed to get summary: ${err.message}`);
    }
}
