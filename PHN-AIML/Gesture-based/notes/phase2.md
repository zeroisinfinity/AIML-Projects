## **Phase 2 — Latency Breakdown**

---

## **1. What is Latency (Real Meaning)**

Latency is not “delay” in a vague sense.

It is:

$$
L = t_{output} - t_{input}
$$

Meaning:

- you move your finger at time $t$
- you hear the effect at time $t + L$

---

## **2. Why Latency Matters Here**

If:

$$
L > 200\text{ ms}
$$

System feels:

- disconnected  
- laggy  
- unusable for control  

If:

$$
L < 50\text{ ms}
$$

System feels:

- responsive  
- instrument-like  

---

## **3. Latency Decomposition**

Total latency is not one thing.

It is sum of parts:

$$
L = L_{capture} + L_{process} + L_{buffer} + L_{output}
$$

---

## **4. Your Current Situation (Hidden Problem)**

From your setup:

- chunk = 4096  
- sample rate = 22050  

$$
L_{buffer} = \frac{4096}{22050} \approx 185\text{ ms}
$$

That alone already breaks the system.

Even if processing is instant, you are still slow.

---

## **Key Insight**

Latency is dominated by:

$$
\text{Chunk Size}
$$

Not just DSP.

---

## **Conclusion**

Before optimizing algorithms, we must:

- control chunk size  
- control buffering  
- define latency budget  

---

Next: we break each latency component and see where we can actually reduce it.

---

## **Phase 2.1 — Latency Components (Quantified)**

---

## **1. Expand the Equation**

$$
L = L_{capture} + L_{process} + L_{buffer} + L_{output}
$$

We estimate each term.

---

## **2. Capture Latency**

Camera frame rate:

$$
f_{cam} \approx 30 \text{ FPS}
$$

So:

$$
L_{capture} \approx \frac{1}{30} \approx 33\text{ ms}
$$

You cannot beat this without a faster camera.

---

## **3. Buffer Latency**

Chunk size dominates:

$$
L_{buffer} = \frac{N}{f_s}
$$

If:

- $N = 4096$
- $f_s = 22050$

$$
L_{buffer} \approx 185\text{ ms}
$$

This is already too high.

---

## **4. Processing Latency**

With good backend:

$$
L_{process} \approx 5\text{–}20\text{ ms}
$$

With librosa:

$$
L_{process} \approx 100\text{–}200\text{ ms}
$$

You already fixed this part.

---

## **5. Output Latency**

Audio driver buffering:

$$
L_{output} \approx 10\text{–}30\text{ ms}
$$

Depends on system, not fully controllable.

---

## **6. Total (Your Current System)**

$$
L \approx 33 + 185 + 20 + 20 = 258\text{ ms}
$$

That explains everything.

---

## **7. Target System**

We want:

$$
L < 80\text{ ms}
$$

So we must reduce:

$$
L_{buffer}
$$

---

## **8. What That Implies**

Solve:

$$
\frac{N}{22050} < 40\text{ ms}
$$

$$
N < 882
$$

So ideal chunk size:

$$
N \approx 512 \text{ or } 1024
$$

---

## **Critical Insight**

Your system is slow not because of DSP anymore.

It is slow because:

$$
\text{Chunk Size} \gg \text{Real-Time Requirement}
$$

---

## **Conclusion**

To reach instrument-level responsiveness:

- Reduce chunk size drastically  
- Keep processing fast  
- Maintain stability with buffering  

---

Next: we solve the main contradiction

**Small chunk = fast but unstable**  
**Large chunk = stable but slow**

---

## **Phase 2.2 — The Core Contradiction**

---

## **1. The Fundamental Tradeoff**

You are trapped between two constraints:

$$
\text{Small Chunk} \Rightarrow \text{Low Latency but Unstable}
$$

$$
\text{Large Chunk} \Rightarrow \text{Stable but High Latency}
$$

---

## **2. Why Small Chunk Fails**

Let:

$$
N = 512 \Rightarrow T_{chunk} \approx 23\text{ ms}
$$

Now system must satisfy:

$$
T_{process} \leq 23\text{ ms}
$$

If processing spikes even once:\
Audio thread → NEED DATA\
Processing → still computing


Result:

- underrun  
- glitch  
- click  

---

## **3. Why Large Chunk “Works”**

Let:

$$
N = 16384 \Rightarrow T_{chunk} \approx 743\text{ ms}
$$

Now:

$$
T_{process} \ll T_{chunk}
$$

So system never starves.

But:

$$
L_{buffer} \uparrow \Rightarrow \text{Responsiveness} \downarrow
$$

---

## **4. Why You Cannot Solve This Directly**

You cannot make:

$$
T_{process} = 0
$$

And you cannot make:

$$
N = 0
$$

So direct approach fails.

---

## **5. The Correct Strategy**

You decouple:

$$
\text{Processing Rate} \neq \text{Playback Rate}
$$

---

## **6. Concept: Producer–Consumer System**

Instead of:\
process → play immediately

We do:\
process → queue → play


---

### **ASCII Model**

Producer (DSP):\
[chunk1] , [chunk2] , [chunk3]....

Queue (buffer):\
|chunk1|chunk2|chunk3|

Consumer (audio output):\
→ → → continuous playback\


---

## **7. Mathematical View**

Let:

- $r_p$ = production rate  
- $r_c$ = consumption rate  

We need:

$$
r_p \geq r_c \quad \text{on average}
$$

Not necessarily at every instant.

---

## **8. What This Solves**

- short delays in processing → absorbed by queue  
- playback never starves  
- chunk size can be small  

---

## **9. New System Goal**

Instead of:

$$
T_{process} \leq T_{chunk}
$$

We relax to:

$$
\mathbb{E}[T_{process}] \leq T_{chunk}
$$

Expected value, not strict per-cycle constraint.

---

## **10. Key Insight**

Stability does not require:

- instant processing  

It requires:

- buffered continuity  

---

## **Conclusion**

We solve the contradiction by introducing:

$$
\text{Buffer Queue + Asynchronous Processing}
$$

This allows:

- small chunks  
- low latency  
- stable output  

---

Next: we design the **buffer mathematically and structurally**


---

## **Phase 2.3 — Buffer Design (Queue Mathematics)**

---

## **1. What the Buffer Actually Is**

A buffer is a finite queue:

$$
Q = \{x_1, x_2, \dots, x_K\}
$$

Where:

- $K$ = capacity (number of chunks stored)
- each element = one processed chunk

---

## **2. State of the System**

At any time:

$$
|Q(t)| = \text{number of ready chunks}
$$

---

## **3. Failure Conditions**

### Underrun (what you had before)

$$
|Q(t)| = 0
$$


Consumer → wants data Queue → empty


Result:
- silence
- stutter

---

### Overrun (next problem you’ll hit)

$$
|Q(t)| = K
$$

Producer → wants to add
Queue → full


Result:
- delay growth
- latency explosion

---

## **4. Stable Operating Region**

We want:

$$
1 \leq |Q(t)| \leq K-1
$$

Meaning:
- never empty  
- never full  

---

## **5. Latency Contribution of Queue**

Each chunk adds delay:

$$
L_{queue} = |Q| \cdot T_{chunk}
$$

Example:

- $|Q| = 3$
- $T_{chunk} = 23\text{ ms}$

$$
L_{queue} \approx 69\text{ ms}
$$

---

## **6. Design Target**

We don’t want big queues.

We want:

$$
|Q| \approx 2 \text{ or } 3
$$

So:

- enough safety margin  
- minimal latency  

---

## **7. Dynamic Behavior**

Producer is not constant:

$$
T_{process} = \text{variable}
$$

So queue evolves:

$$
|Q(t+1)| = |Q(t)| + r_p - r_c
$$

Where:

- $r_p$ = produced chunks per unit time  
- $r_c$ = consumed chunks per unit time  

---

## **8. Control Objective**

Keep:

$$
|Q(t)| \rightarrow Q_{target}
$$

Where:

$$
Q_{target} \approx 2
$$

---

## **9. Practical Policy**

We enforce:

- if queue empty → reuse last chunk  
- if queue too large → drop oldest  

---

### ASCII Flow
Producer → [C1][C2][C3] → Queue → Consumer

If slow:
Queue shrinks → reuse last

If fast:
Queue grows → drop old


---

## **10. Critical Insight**

The buffer is not storage.

It is a **stability controller**.

---

## **Conclusion**

You now have:

- bounded latency  
- no starvation  
- tolerance to processing spikes  

---

Next: we map this into an actual system architecture

threads + data flow + timing relationships