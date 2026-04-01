## **7. Gesture → Control Mapping (The Transformation Model)**

This module formalizes the translation of **Computer Vision (Spatial)** data into **Audio DSP (Temporal)** parameters. We define this as a linear mapping function.

### **7.1 The Fundamental Mapping Equation**
For any input gesture $g(t) \in [0,1]$, the target control parameter $\theta(t)$ is calculated as:

$$\theta(t) = \underbrace{\alpha \cdot g(t)}_{\text{Scaling (Range)}} + \underbrace{\beta}_{\text{Translation (Offset)}}$$

#### **Mathematical Variable Meanings:**
*   **$\theta(t)$ (Control Parameter):** The physical unit sent to the audio engine (e.g., Semitones, Multipliers).
*   **$g(t)$ (Gesture Input):** The normalized coordinate from the camera $[0.0, 1.0]$.
*   **$\alpha$ (Gain/Slope):** The **Total Range** of the effect ($\text{Max} - \text{Min}$). This determines sensitivity.
*   **$\beta$ (Bias/Intercept):** The **Starting Point** value when the gesture input is zero.

---

### **7.2 Full Parameter Formulas**


| Effect | Symbol ($\theta$) | Input ($g$) | Range ($\alpha$) | Offset ($\beta$) | **The Full Transformation** |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pitch** | $p$ | $y_{index}$ | $-24$ | $+12$ | $p(t) = (-24 \cdot y_{idx}) + 12$ |
| **Speed** | $s$ | $x_{thumb}$ | $+1.5$ | $+0.5$ | $s(t) = (1.5 \cdot x_{thm}) + 0.5$ |
| **Volume**| $v$ | $y_{mid}$ | $-1.0$ | $+1.0$ | $v(t) = (-1.0 \cdot y_{mid}) + 1.0$ |

---

### **7.3 Scenario Analysis & Logic**

#### **Scenario A: Pitch Shift ($p$)**
*   **Goal:** Map vertical hand position to $\pm 1$ Octave (-12 to +12 semitones).
*   **Math:** $p(t) = (-24 \cdot 0) + 12 = \mathbf{+12}$ (High). 
*   **Logic:** $\alpha = -24$ because as $y$ increases (hand moves down), $p$ must decrease.

#### **Scenario B: Playback Speed ($s$)**
*   **Goal:** Map horizontal position to $0.5\times$ (Slow) to $2.0\times$ (Fast).
*   **Math:** $s(t) = (1.5 \cdot 1) + 0.5 = \mathbf{2.0x}$.
*   **Logic:** $\beta = 0.5$ (minimum speed). $\alpha = 1.5$ (the "distance" from 0.5 to 2.0).

#### **Scenario C: Amplitude Gain ($v$)**
*   **Goal:** Use finger height as a volume slider (0.0 to 1.0).
*   **Math:** $v(t) = (-1.0 \cdot 1) + 1.0 = \mathbf{0.0}$ (Muted).
*   **Logic:** $\alpha = -1.0$ inverts the axis so "Up" is "Loud" and "Down" is "Silent."

---

### **7.4 Stability Filter (EMA)**
To prevent spectral clicking/popping caused by camera jitter, we apply a **First-Order Recursive Filter**:

$$\theta_{stable}[n] = \lambda \cdot \theta_{raw} + (1 - \lambda) \cdot \theta_{stable}[n-1]$$

*Where $\lambda$ (Smoothing Factor) typically equals $0.1$.*

---

## **10. DSP Transformations (The Signal Processing Engine)**

This section defines the mathematical operator $\mathcal{T}_{\theta}$ that transforms the input digital signal $x[n]$ into the modulated output $y[n]$ in real-time.

---

### **10.1 The Transformation Operator**
We represent the entire system as a functional mapping where the output is a result of the transformation $\mathcal{T}$ applied to the input sequence, governed by the control parameter vector $\theta$:

$$y[n] = \mathcal{T}_{\theta}(x[n])$$

### **10.2 Mathematical DSP Models**

#### **I. Pitch Shift (Resampling Model)**
To shift pitch, we manipulate the **phase increment** of the signal. In a discrete system, this is achieved by modifying the rate at which we traverse the samples.

$$y[n] = x\left(n \cdot 2^{\frac{p}{12}}\right)$$

*   **The Exponent ($p/12$):** Since Western music uses **12-Tone Equal Temperament**, an octave is a doubling of frequency ($2^1$). A single semitone is the $12^{th}$ root of 2 ($2^{1/12}$).
*   **The Logic:** 
    *   If $p = +12$ (One octave up), the ratio is $2^1 = 2$. We read every **2nd** sample, "squashing" the wave.
    *   If $p = -12$ (One octave down), the ratio is $2^{-1} = 0.5$. We read every **half** sample (interpolating), "stretching" the wave.

---

#### **II. Speed Change (Time Scaling)**
Speed modulation determines the velocity of the read-pointer through the audio buffer without necessarily altering the pitch (in advanced Phase Vocoders).

$$y[n] = x(n \cdot s)$$

*   **The Parameter ($s$):** A linear scalar for the discrete index $n$.
*   **The Logic:** 
    *   $s > 1.0$: The system skips indices to move forward faster (**Fast Forward**).
    *   $s < 1.0$: The system repeats or interpolates indices to move slower (**Slow Motion**).

---

#### **III. Volume (Amplitude Scaling)**
The simplest DSP operation, volume modulation is a linear multiplication of the instantaneous amplitude of each sample.

$$y[n] = v \cdot x[n]$$

*   **The Scalar ($v$):** Represented as a gain factor $v \in [0.0, 1.0]$.
*   **The Logic:** This scales the **Vertical Displacement** of the waveform.
    *   If $v = 0.5$, the wave is half as tall ($-6\text{dB}$ gain).
    *   If $v = 0.0$, the result is a "Zero-Vector" (Total Silence).

---

### **10.3 Summary Table for Engineering Implementation**


| Transformation | Mathematical Intuition | Visual Waveform Effect |
| :--- | :--- | :--- |
| **Pitch ($p$)** | **Exponential Resampling:** $\Delta n \propto 2^{p/12}$ | Horizontal Compression/Expansion |
| **Speed ($s$)** | **Linear Index Scaling:** $\Delta n \propto s$ | Temporal Duration Change |
| **Volume ($v$)**| **Amplitude Gain:** $y \propto v \cdot x$ | Vertical Height Scaling |

---

### **10.4 The Computational Reality**
In a system where $f_s = 44,100\text{Hz}$, the DSP engine must execute these multiplications **44,100 times per second** for every active channel. This requires the **Real-Time Constraint** ($T_{process} \leq T_{chunk}$) to be strictly maintained to avoid buffer underruns.

---