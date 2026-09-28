/* -------------------------------------------------------------
   VR Live Captions & MoM AI - Frontend Application Logic
   ------------------------------------------------------------- */

document.addEventListener('DOMContentLoaded', () => {
    // Nav Tab Buttons
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // FEATURE 1: VR Live Elements
    const vrTargetLangSelect = document.getElementById('vrTargetLangSelect');
    const vrMicDeviceSelect = document.getElementById('vrMicDeviceSelect');
    const vrLiveToggleBtn = document.getElementById('vrLiveToggleBtn');
    const vrLiveStatusBadge = document.getElementById('vrLiveStatusBadge');
    
    const vrActiveSpeaker = document.getElementById('vrActiveSpeaker');
    const vrTranslatedText = document.getElementById('vrTranslatedText');
    const vrOriginalText = document.getElementById('vrOriginalText');
    const vrLatency = document.getElementById('vrLatency');
    const vrSourceLang = document.getElementById('vrSourceLang');
    const vrTargetLangText = document.getElementById('vrTargetLangText');
    const vrStreamLog = document.getElementById('vrStreamLog');

    // FEATURE 2: Meeting & MoM Elements
    const meetingInputModeRadios = document.querySelectorAll('input[name="meetingInputMode"]');
    const meetingFileUploadContainer = document.getElementById('meetingFileUploadContainer');
    const meetingMicSelectBox = document.getElementById('meetingMicSelectBox');
    const audioFileInput = document.getElementById('audioFileInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const meetingMicDeviceSelect = document.getElementById('meetingMicDeviceSelect');
    const meetingTargetLangSelect = document.getElementById('meetingTargetLangSelect');

    const meetingStartBtn = document.getElementById('meetingStartBtn');
    const meetingStopBtn = document.getElementById('meetingStopBtn');
    const audioStatusText = document.getElementById('audioStatusText');
    const utteranceCountEl = document.getElementById('utteranceCount');
    const meetingTranscriptFeed = document.getElementById('meetingTranscriptFeed');

    // MoM Studio Elements
    const momResultsStudio = document.getElementById('momResultsStudio');
    const momSessionTitle = document.getElementById('momSessionTitle');
    const momExecutiveSummary = document.getElementById('momExecutiveSummary');
    const momKeyTopics = document.getElementById('momKeyTopics');
    const speakerStatsContainer = document.getElementById('speakerStatsContainer');
    const momActionItemsTable = document.getElementById('momActionItemsTable');
    const diarizationTimeline = document.getElementById('diarizationTimeline');

    // State Variables
    let isVrActive = false;
    let isMeetingActive = false;
    let eventSource = null;
    let selectedFilePath = null;
    let currentSessionId = null;
    let utteranceCounter = 0;
    let animationFrameId = null;

    // ---------------- 1. NAVIGATION TAB SWITCHING ----------------
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const targetTab = document.getElementById(btn.dataset.tab);
            if (targetTab) targetTab.classList.add('active');
        });
    });

    // ---------------- 2. FEATURE 1: REAL-TIME VR LIVE CAPTIONS ----------------
    vrTargetLangSelect.addEventListener('change', async () => {
        const langCode = vrTargetLangSelect.value;
        const langName = vrTargetLangSelect.options[vrTargetLangSelect.selectedIndex].text.replace("Translate to: ", "");
        vrTargetLangText.textContent = `${langName} (${langCode})`;

        if (isVrActive) {
            try {
                await fetch('/api/set_target_language', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lang_code: langCode })
                });
            } catch (err) {
                console.error("Language update error:", err);
            }
        }
    });

    vrLiveToggleBtn.addEventListener('click', async () => {
        if (!isVrActive) {
            await startVrStream();
        } else {
            await stopVrStream();
        }
    });

    async function startVrStream() {
        const targetLang = vrTargetLangSelect.value;
        const deviceIndex = vrMicDeviceSelect.value;

        try {
            connectSSE();

            const resp = await fetch('/api/start_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target_lang: targetLang,
                    device_index: deviceIndex
                })
            });
            const data = await resp.json();

            if (data.status === 'success') {
                isVrActive = true;
                currentSessionId = data.session_id;

                vrLiveToggleBtn.innerHTML = `<i class="fa-solid fa-square"></i> Stop VR Live Stream`;
                vrLiveToggleBtn.className = 'record-btn btn-stop';

                vrLiveStatusBadge.className = 'hud-pill live-pill pulse-red';
                vrLiveStatusBadge.innerHTML = `<span class="dot"></span> LIVE VR STREAM`;

                vrTranslatedText.textContent = "Listening for incoming speech...";
                vrOriginalText.textContent = "";

                startVisualizers();
            }
        } catch (err) {
            console.error("VR Stream error:", err);
            alert("Could not start VR Live Stream.");
        }
    }

    async function stopVrStream() {
        vrLiveToggleBtn.disabled = true;
        try {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }

            await fetch('/api/stop_session', { method: 'POST' });
            
            isVrActive = false;
            vrLiveToggleBtn.disabled = false;
            vrLiveToggleBtn.innerHTML = `<i class="fa-solid fa-play"></i> Start VR Live Stream`;
            vrLiveToggleBtn.className = 'record-btn btn-start';

            vrLiveStatusBadge.className = 'hud-pill live-pill-off';
            vrLiveStatusBadge.innerHTML = `<span class="dot"></span> STANDBY`;

            vrTranslatedText.textContent = "VR Live Stream stopped.";
            stopVisualizers();
        } catch (err) {
            console.error("Stop VR error:", err);
            vrLiveToggleBtn.disabled = false;
        }
    }

    // ---------------- 3. FEATURE 2: MEETING ASSISTANT & MOM AI ----------------
    meetingInputModeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'file') {
                meetingFileUploadContainer.classList.remove('hidden');
                meetingMicSelectBox.classList.add('hidden');
            } else {
                meetingFileUploadContainer.classList.add('hidden');
                meetingMicSelectBox.classList.remove('hidden');
                selectedFilePath = null;
            }
        });
    });

    audioFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        fileNameDisplay.textContent = `Uploading ${file.name}...`;
        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload_audio', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();
            if (data.status === 'success') {
                selectedFilePath = data.filepath;
                fileNameDisplay.textContent = `Ready: ${file.name}`;
                audioStatusText.textContent = "Audio File Uploaded & Ready";
            }
        } catch (err) {
            console.error(err);
            fileNameDisplay.textContent = "Upload Error";
        }
    });

    meetingStartBtn.addEventListener('click', async () => {
        await startMeetingSession();
    });

    meetingStopBtn.addEventListener('click', async () => {
        await stopMeetingSession();
    });

    async function startMeetingSession() {
        const targetLang = meetingTargetLangSelect.value;
        const inputMode = document.querySelector('input[name="meetingInputMode"]:checked').value;
        const deviceIndex = meetingMicDeviceSelect.value;

        const payload = {
            target_lang: targetLang,
            audio_file_path: inputMode === 'file' ? selectedFilePath : null,
            device_index: inputMode === 'mic' ? deviceIndex : null
        };

        try {
            connectSSE();

            const resp = await fetch('/api/start_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();

            if (data.status === 'success') {
                isMeetingActive = true;
                currentSessionId = data.session_id;
                utteranceCounter = 0;
                utteranceCountEl.textContent = '0';

                meetingStartBtn.classList.add('hidden');
                meetingStopBtn.classList.remove('hidden');

                audioStatusText.textContent = inputMode === 'file' ? 'Streaming File Audio...' : 'Live Meeting Recording Active...';
                meetingTranscriptFeed.innerHTML = '';
                momResultsStudio.classList.add('hidden');

                startVisualizers();
            }
        } catch (err) {
            console.error("Start meeting error:", err);
            alert("Could not start meeting session.");
        }
    }

    async function stopMeetingSession() {
        meetingStopBtn.disabled = true;
        meetingStopBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Generating MoM AI...`;

        try {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }

            const resp = await fetch('/api/stop_session', { method: 'POST' });
            const data = await resp.json();

            if (data.status === 'success') {
                isMeetingActive = false;
                meetingStopBtn.disabled = false;
                meetingStopBtn.innerHTML = `<i class="fa-solid fa-square"></i> End Meeting & Generate MoM`;
                meetingStopBtn.classList.add('hidden');
                meetingStartBtn.classList.remove('hidden');

                audioStatusText.textContent = "Meeting Session Finalized";
                stopVisualizers();

                // Show MoM Studio
                renderMoMResults(data.mom);
                momResultsStudio.classList.remove('hidden');
                momResultsStudio.scrollIntoView({ behavior: 'smooth' });
            }
        } catch (err) {
            console.error("Stop meeting error:", err);
            meetingStopBtn.disabled = false;
            meetingStopBtn.innerHTML = `<i class="fa-solid fa-square"></i> End Meeting & Generate MoM`;
        }
    }

    // ---------------- 4. SSE STREAM RECEIVER ----------------
    function connectSSE() {
        if (eventSource) eventSource.close();

        eventSource = new EventSource('/api/stream_captions');
        eventSource.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload.type === 'caption') {
                    handleIncomingCaption(payload.data);
                }
            } catch (e) {
                console.error("SSE parse error:", e);
            }
        };

        eventSource.onerror = (err) => {
            console.warn("SSE reconnecting...", err);
        };
    }

    function handleIncomingCaption(item) {
        utteranceCounter++;
        utteranceCountEl.textContent = utteranceCounter.toString();

        // Update Feature 1 (VR HUD View)
        if (vrActiveSpeaker) vrActiveSpeaker.innerHTML = `<i class="fa-solid fa-user-astronaut"></i> ${item.speaker}`;
        if (vrTranslatedText) vrTranslatedText.textContent = `"${item.translated_text}"`;
        if (vrOriginalText) vrOriginalText.textContent = `[${item.source_lang.toUpperCase()}] ${item.original_text}`;
        if (vrSourceLang) vrSourceLang.textContent = item.source_lang.toUpperCase();
        if (vrLatency) vrLatency.textContent = `${item.latency_ms || 220}ms`;

        if (vrStreamLog) {
            const logDiv = document.createElement('div');
            logDiv.className = 'log-item';
            logDiv.innerHTML = `<strong>${item.speaker}:</strong> ${item.translated_text}`;
            vrStreamLog.prepend(logDiv);
        }

        // Update Feature 2 (Meeting Transcript Feed)
        if (meetingTranscriptFeed) {
            const feedItem = document.createElement('div');
            feedItem.className = 'transcript-item';
            feedItem.innerHTML = `
                <div class="item-meta">
                    <span class="speaker-badge"><i class="fa-solid fa-circle-user"></i> ${item.speaker}</span>
                    <span class="time-stamp">${item.timestamp} (${item.latency_ms || 220}ms)</span>
                </div>
                <div class="translated-body">${item.translated_text}</div>
                <div class="original-body">[${item.source_lang.toUpperCase()}] ${item.original_text}</div>
            `;
            meetingTranscriptFeed.prepend(feedItem);
        }
    }

    // ---------------- 5. AUDIO SPECTRUM VISUALIZERS ----------------
    const waveformCanvas = document.getElementById('waveformCanvas');
    const waveformCtx = waveformCanvas ? waveformCanvas.getContext('2d') : null;
    const radarCanvas = document.getElementById('radarCanvas');
    const radarCtx = radarCanvas ? radarCanvas.getContext('2d') : null;

    function startVisualizers() {
        let phase = 0;
        function draw() {
            if (!isVrActive && !isMeetingActive) return;

            if (waveformCtx) {
                waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
                waveformCtx.beginPath();
                waveformCtx.lineWidth = 2;
                waveformCtx.strokeStyle = '#38bdf8';

                const sliceWidth = waveformCanvas.width / 50;
                let x = 0;
                for (let i = 0; i < 50; i++) {
                    const v = Math.sin(phase + i * 0.2) * 18 + 35;
                    if (i === 0) waveformCtx.moveTo(x, v);
                    else waveformCtx.lineTo(x, v);
                    x += sliceWidth;
                }
                waveformCtx.stroke();
            }

            if (radarCtx) {
                radarCtx.clearRect(0, 0, radarCanvas.width, radarCanvas.height);
                radarCtx.fillStyle = 'rgba(56, 189, 248, 0.15)';
                radarCtx.beginPath();
                radarCtx.arc(60, 30, 20 + Math.sin(phase * 2) * 5, 0, Math.PI * 2);
                radarCtx.fill();
            }

            phase += 0.1;
            animationFrameId = requestAnimationFrame(draw);
        }
        draw();
    }

    function stopVisualizers() {
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
        if (waveformCtx) waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        if (radarCtx) radarCtx.clearRect(0, 0, radarCanvas.width, radarCanvas.height);
    }

    // ---------------- 6. RENDER POST-MEETING MOM RESULTS ----------------
    function renderMoMResults(mom) {
        momSessionTitle.textContent = `Minutes of Meeting (${mom.session_id})`;
        momExecutiveSummary.textContent = mom.executive_summary || "No summary available.";

        momKeyTopics.innerHTML = (mom.key_topics || []).map(t => `<span class="topic-tag">#${t}</span>`).join('') || '<span class="topic-tag">#GeneralDiscussion</span>';

        const speakers = mom.speaker_analytics?.speakers || [];
        speakerStatsContainer.innerHTML = speakers.map(s => `
            <div class="speaker-stat-item">
                <div class="stat-header">
                    <span><strong>${s.speaker}</strong> (${s.total_words} words, ${s.turns_count} turns)</span>
                    <span>${s.percentage}% (${s.duration_sec}s)</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: ${s.percentage}%"></div>
                </div>
            </div>
        `).join('') || '<p class="text-muted">No speaker data recorded.</p>';

        const actions = mom.action_items || [];
        if (actions.length > 0) {
            momActionItemsTable.innerHTML = actions.map(a => `
                <tr>
                    <td><code>${a.timestamp}</code></td>
                    <td><strong>${a.assignee}</strong></td>
                    <td>${a.task}</td>
                    <td><span class="status-badge">${a.status}</span></td>
                </tr>
            `).join('');
        } else {
            momActionItemsTable.innerHTML = `<tr><td colspan="4" class="text-center text-muted">No pending action items extracted.</td></tr>`;
        }

        const timeline = mom.speaker_analytics?.timeline || [];
        const maxDuration = mom.estimated_duration_sec || 1;
        
        if (timeline.length > 0) {
            diarizationTimeline.innerHTML = timeline.map(t => {
                const leftPct = ((t.start / maxDuration) * 100).toFixed(1);
                const widthPct = (Math.max(2, (t.end - t.start) / maxDuration * 100)).toFixed(1);
                return `
                    <div class="timeline-row">
                        <div class="timeline-speaker">${t.speaker}</div>
                        <div class="timeline-track">
                            <div class="timeline-block" style="left: ${leftPct}%; width: ${widthPct}%;" title="${t.text}"></div>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            diarizationTimeline.innerHTML = `<div class="empty-state">No timeline events captured.</div>`;
        }
    }

    // Export Handlers
    document.getElementById('exportHtmlBtn').addEventListener('click', () => {
        if (currentSessionId) window.location.href = `/api/download_mom/${currentSessionId}/html`;
    });
    document.getElementById('exportMdBtn').addEventListener('click', () => {
        if (currentSessionId) window.location.href = `/api/download_mom/${currentSessionId}/md`;
    });
    document.getElementById('exportJsonBtn').addEventListener('click', () => {
        if (currentSessionId) window.location.href = `/api/download_mom/${currentSessionId}/json`;
    });
});
