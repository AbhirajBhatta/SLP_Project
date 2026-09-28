# VR Application Integration Blueprint (Unity / Cloud Streaming)

This guide documents how to connect a **Virtual Reality HMD application (Unity / Unreal Engine / WebXR)** to the Real-Time Captions & Translation Backend Server via WebSockets / SSE.

---

## 1. Architecture Overview

```
+------------------------------------+
|        VR Headset (Unity C#)       |
|  - Meta Quest 3 / Apple Vision Pro |
|  - Renders 3D Curved Subtitle Canvas|
+------------------------------------+
                   ^
                   | WebSockets / SSE Stream (JSON)
                   v
+------------------------------------+
|    Python SLP Backend Server       |
|  - ASR (Faster-Whisper CUDA)       |
|  - Neural Machine Translation      |
|  - Real-Time Speaker Diarization   |
+------------------------------------+
```

---

## 2. Unity C# WebSocket Subtitle Receiver Script

Add the following C# script (`VRCaptionReceiver.cs`) to a World-Space Canvas in your Unity VR scene (attached to a curved floating UI panel anchored in camera view):

```csharp
using System;
using System.Collections;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using TMPro;

[Serializable]
public class CaptionData
{
    public string speaker;
    public string original_text;
    public string translated_text;
    public string source_lang;
    public string target_lang;
    public int latency_ms;
}

[Serializable]
public class SSEResponse
{
    public string type;
    public CaptionData data;
}

public class VRCaptionReceiver : MonoBehaviour
{
    [Header("Server Settings")]
    public string serverUrl = "ws://127.0.0.1:5000/api/ws_captions"; // or SSE HTTP endpoint

    [Header("VR Subtitle Canvas UI")]
    public TextMeshProUGUI speakerText;
    public TextMeshProUGUI translatedText;
    public TextMeshProUGUI originalText;
    public TextMeshProUGUI latencyText;

    private ClientWebSocket webSocket;
    private CancellationTokenSource cancellationTokenSource;

    private void Start()
    {
        cancellationTokenSource = new CancellationTokenSource();
        Task.Run(() => ConnectAndReceive(cancellationTokenSource.Token));
    }

    private async Task ConnectAndReceive(CancellationToken token)
    {
        webSocket = new ClientWebSocket();
        try
        {
            Uri serverUri = new Uri(serverUrl);
            await webSocket.ConnectAsync(serverUri, token);
            Debug.Log("[VR Subtitles] Connected to SLP Backend Stream!");

            byte[] buffer = new byte[4096];
            while (webSocket.State == WebSocketState.Open && !token.IsCancellationRequested)
            {
                WebSocketReceiveResult result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
                string jsonString = Encoding.UTF8.GetString(buffer, 0, result.Count);
                
                // Parse JSON payload
                SSEResponse response = JsonUtility.FromJson<SSEResponse>(jsonString);
                if (response != null && response.type == "caption")
                {
                    // Dispatch to Unity Main Thread for UI rendering
                    UnityMainThreadDispatcher.Instance().Enqueue(() => UpdateVRHUD(response.data));
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[VR Subtitles] Connection error: {ex.Message}");
        }
    }

    private void UpdateVRHUD(CaptionData data)
    {
        if (speakerText != null) speakerText.text = $"<color=#38bdf8>{data.speaker}</color>";
        if (translatedText != null) translatedText.text = $"\"{data.translated_text}\"";
        if (originalText != null) originalText.text = $"[{data.source_lang.ToUpper()}] {data.original_text}";
        if (latencyText != null) latencyText.text = $"⚡ {data.latency_ms}ms";
    }

    private void OnDestroy()
    {
        cancellationTokenSource?.Cancel();
        webSocket?.Dispose();
    }
}
```

---

## 3. Spatial Audio & Directional Subtitle Placement

To anchor subtitles above the active speaker's avatar head in 3D VR space:
1. Match the `speaker` tag (`Speaker 1`, `Speaker 2`) with participant avatars in your Unity scene.
2. Position the `TextMeshPro` 3D Canvas above `avatarHead.transform.position + Vector3.up * 0.3f`.
3. Apply `transform.LookAt(Camera.main.transform)` so subtitles face the user regardless of head rotation.
