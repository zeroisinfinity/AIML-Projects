## **3. Latency Decomposition (The Temporal Budget)**

System latency $L$ is not a single delay, but an accumulation of sequential bottlenecks across the hardware-software boundary. To maintain the **Real-Time Constraint**, we must satisfy the condition:  
$$L_{total} \leq 50\text{ ms}$$

---

### **3.1 The Latency Summation Model**
The total system latency is defined as the sum of its discrete components:

$$L = L_{capture} + L_{process} + L_{buffer} + L_{output}$$

#### **Mathematical Component Breakdown:**

1.  **$L_{capture}$ (Input Lag):**  
    The time taken for the camera/microphone to digitize the physical event.  
    $$L_{capture} = \frac{1}{\text{FPS}_{cam}}$$  
    *Example: At 30 FPS, $L_{capture} \approx 33.3\text{ ms}$.*

2.  **$L_{process}$ (Computational Cost):**  
    The CPU time required to run Hand-Tracking (Inference) and DSP calculations.  
    $$L_{process} = T_{inference} + T_{DSP}$$

3.  **$L_{buffer}$ (Algorithmic Delay):**  
    The time audio sits in the "waiting room" before being processed. This is tied to the **Chunk Size ($N$)**.  
    $$L_{buffer} = \frac{N}{f_s}$$  
    *Example: $N=1024, f_s=44100 \Rightarrow L_{buffer} \approx 23.2\text{ ms}$.*

4.  **$L_{output}$ (Hardware DAC):**  
    The time taken for the Digital-to-Analog Converter (DAC) to push the signal to the speakers.

---

### **3.2 Latency Scenarios & Impact**


| Scenario | $L_{total}$ Value | Perceptual Result | Engineering Verdict |
| :--- | :--- | :--- | :--- |
| **Instantaneous** | $< 10\text{ ms}$ | Imperceptible | **Gold Standard** (Requires ASIO/C++) |
| **Musical** | $10\text{--}40\text{ ms}$ | "Tight" response | **Project Goal** (Acceptable for Gestures) |
| **Sluggish** | $50\text{--}100\text{ ms}$ | Noticable "Drag" | **Sub-optimal** (Frustrating for User) |
| **Broken** | $> 150\text{ ms}$ | Disconnected feel | **Failure** (System Collapse) |

---

### **3.3 The Real-Time Constraint Formula**
To prevent **Buffer Underrun** (stuttering/glitching), the processing time must always be less than the duration of the audio chunk:

$$T_{process} \leq \frac{N}{f_s}$$

If this inequality fails, the audio thread starves for data, resulting in a **Spectral Discontinuity** (Audible Pop).

---

### **3.4 Summary for Implementation**
*   **To reduce $L$:** We must decrease $N$ (Chunk Size).
*   **The Trade-off:** Smaller $N$ gives lower latency but increases the risk of CPU spikes causing audio glitches.
