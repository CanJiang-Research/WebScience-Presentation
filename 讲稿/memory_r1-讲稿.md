# Presentation Script (20 Minutes): *Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning*

**Course / Event:** Web Science Literature Report, Southeast University (Group 14)  
**Paper:** Sikuan Yan et al., *Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning* (ACL 2026, pp. 12805–12825)  
**Target Duration:** ~20 minutes (~130 words/minute clear academic speaking pace, including slide transitions and speaker handoffs)  
**Presenters:** Presenter 1 (Background, Slides 1–3) | Presenter 2 (Method, Slides 4–9) | Presenter 3 (Experiments & Discussion, Slides 10–15)

## Time Allocation Overview

| Section | Presenter | Slides | Target Time | Key Focus |
| :--- | :--- | :--- | :--- | :--- |
| Background & Related Work | Presenter 1 | Slides 1–3 | 4.5 min | Opening hook, Web Science knowledge engineering framing, three bottlenecks, two-adoption example, positioning |
| Method & RL Design | Presenter 2 | Slides 4–9 | 8.0 min | Two-policy architecture, six read/write operations, worked dialogue trace, snapshot construction, PPO vs. GRPO, binary EM reward |
| Experiments, Limitations & Discussion | Presenter 3 | Slides 10–15 | 7.5 min | Setup & LoCoMo overall results, per-category trade-offs, ablations, reward curves & reward design, transfer/scaling, Web Science directions & Q&A prompt |
| **Total** | **3 Presenters** | **15 Slides** | **20.0 min** | **~2,600 spoken words, balanced pacing** |

## Part 1: Research Background & Related Work (Presenter 1 · ~4.5 min)

### Slide 1: Title & Opening (~1.0 min)

> **[Delivery Cue]** Stand centered, make eye contact with the audience, and open at a steady, engaging pace before introducing the paper metadata.

Good afternoon, professors and classmates. We are Group 14. Today, we are presenting our Web Science literature report on the ACL 2026 paper titled *Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning*, by Sikuan Yan and colleagues.

Before we dive into equations and tables, consider a familiar failure mode in multi-session conversations: a dialogue history can contain every piece of information needed to answer a question, and yet an LLM agent still gets the answer wrong. Often, the problem is not that the agent missed the fact originally—it is that the agent overwrote a valid memory earlier, or got distracted by noisy entries when reading its memory bank later.

Over the next twenty minutes, the three of us will trace one concrete example through three parts: first, the research background and related work; second, Memory-R1's two-policy reinforcement learning framework; and third, the experimental findings, limitations, and Web Science connections. Let's begin with the background.

### Slide 2: 1 Research Background and Significance (~2.0 min)

> **[Delivery Cue]** Gesture to the left workflow ("BUILD → MAINTAIN → USE"), then move to the three bottlenecks in the center, and walk carefully through diagrams (a), (b), and (c) on the right.

From a Web Science perspective, long-term agent memory maps directly onto the classic knowledge engineering lifecycle: building knowledge, maintaining it over time, and using it when needed. In an agent's external memory bank, building corresponds to adding new facts, maintaining corresponds to updating or deleting entries, and using corresponds to retrieving relevant records at query time.

However, in real multi-session interactions, records evolve. A single new fact in a later session can change what a future question requires. This creates three fundamental bottlenecks:

First, **writing across sessions**. When a new fact arrives, does it contradict and replace an earlier memory, or does it extend it? Text similarity alone cannot resolve that distinction.  
Second, **reading under noise**. Retrieval can return dozens of candidate memories, and the agent must separate true supporting evidence from distractors.  
Third, **sparse supervision**. We rarely have step-by-step labels telling an agent which memory edit to make. Instead, feedback arrives much later, only through whether the final answer is right or wrong.

To see this concretely, look at the two-adoption scenario on the right, which runs throughout the paper:  
In panel (a), Andrew mentions in an early session that he adopted a shelter dog named Buddy. In a later session, he mentions adopting *another* dog named Scout. Notice how important the word "another" is—Buddy is still part of the record, so Andrew has adopted two dogs in total.

Now look at panel (b), a prompted memory manager. When the new fact about Scout queries the memory bank, it retrieves the entry about Buddy. A prompted model can easily misinterpret this as a conflict, dropping Buddy and adding Scout via `DELETE + ADD`. Later, when asked "How many dogs has Andrew adopted?", even retrieving up to 60 candidate memories only surfaces Scout—so the agent wrongly answers "1 dog."

In panel (c), **Memory-R1** learns via reinforcement learning to execute `UPDATE` instead, extending the entry to preserve Buddy alongside Scout. At question time, the Answer Agent selects that consolidated record and correctly answers "2 dogs."

Summarizing the significance at the bottom right: Memory-R1 learns both how to write and how to read memory; it trains both policies using PPO or GRPO from just 152 question–answer pairs in LoCoMo; and it evaluates transfer beyond LoCoMo onto MSC and LongMemEval.

### Slide 3: 2 Related Work (~1.5 min)

> **[Delivery Cue]** Contrast top-left ("External Memory Systems") and bottom-right ("Evidence Selection"), highlight the Search-R1 inspiration at top-right, and hand off smoothly to Presenter 2.

Slide 3 positions Memory-R1 along two lines of prior work plus one key methodological inspiration.

On the top left, **external memory systems** aim to maintain persistent knowledge across interactions—for example, ReadAgent stores gist memories, A-Mem links structured entries, Mem0 prompts the model to edit records, and MemoryOS organizes memories across sessions. While these systems design rich storage structures, they still rely on hand-crafted prompts to decide edits, which can discard facts that still hold—and a single memory entry that helps one query can easily distract from another.

On the bottom right, **evidence selection** approaches such as RAG and reranking retrieve and reorder candidate chunks for the current question. Here, the trade-off is sharp: retrieving too few candidates risks missing a necessary fact, while retrieving too many buries the useful evidence in noise.

The key inspiration comes from **Search-R1** on the top right, which showed that reinforcement learning can train intermediate search decisions using only the correctness of the final answer. Yet memory management is strictly harder than search: a search action only fetches evidence for the current task, whereas a memory edit mutates the external store and changes the evidence available to *all future tasks*.

Therefore, as summarized in the bottom left, Memory-R1 moves beyond prompted rules by training two RL policies: one to learn which memory operations lead to correct future answers, and a second to select supporting evidence and generate the response.

I will now hand over to Presenter 2 to walk us through the architecture and reinforcement learning design.

## Part 2: Method & Reinforcement Learning Design (Presenter 2 · ~8.0 min)

### Slide 4: 3.1 The Memory-R1 Architecture (~1.2 min)

> **[Delivery Cue]** Point to the central feedback loop connecting Policy Inputs, Memory-R1, and Memory Bank $M$, then map it onto steps 01–04 on the right.

Thank you. I will now walk through how Memory-R1 is structured and trained.

At a high level, Memory-R1 decomposes long-term memory into **two separately parameterized policies**—a Memory Manager ($\pi_{\text{write}}$) and an Answer Agent ($\pi_{\text{answer}}$)—which interact exclusively through an external memory bank $M$.

Following the four steps on the right:  
- **Step 01 (Manager Input):** At write time, a newly observed dialogue fact $x$ acts as the query to retrieve related existing entries $M_{\text{old}}$, forming the manager's state $s = (x, M_{\text{old}})$.  
- **Step 02 (Write Policy):** Given $s$, the write policy chooses one edit operation $o$ from `{ADD, UPDATE, DELETE, NOOP}` and applies it to produce an updated memory bank $M'$.  
- **Step 03 (Read Policy):** At read time, a question $q$ retrieves a set of candidate entries $M_{\text{ret}}$, and the Answer Agent distills the relevant evidence before generating an answer $y$.  
- **Step 04 (Reward):** Finally, the generated answer $y$ is compared against the gold reference $y^*$ using Exact Match to produce the reward $r$.

A key design choice noted at the bottom is that the two agents are **trained separately**: while one policy is being updated, its partner stays frozen. This decouples the learning signal and stabilizes training.

### Slide 5: 3.2 Memory Operations and Distillation (~1.3 min)

> **[Delivery Cue]** Distinguish the top four green rows (write path) from the bottom two beige rows (read path), then point to the three-box flow at the bottom right.

Slide 5 breaks down the six operations across the write and read paths.

The first four rows belong to the **write path**, executed by the Memory Manager on the memory bank:  
- **ADD** appends a brand-new persistent record for a new fact.  
- **UPDATE** revises or merges an existing entry with the new fact—for instance, in the bottom-left example, updating the entry for Andrew's dogs to read "Andrew adopted Buddy and Scout."  
- **DELETE** removes an entry that has become obsolete.  
- **NOOP** is an explicit "do nothing" choice when existing entries already capture the information, preventing unnecessary edits from corrupting the bank.

The final two rows belong to the **read path**:  
- **RETRIEVE** performs similarity search for a question and returns up to 60 candidate records.  
- **DISTILL** is the Answer Agent's internal evidence-selection step: instead of answering blindly from all 60 candidates, it identifies the subset of relevant entries and generates the answer from that distilled evidence.

As the diagram at the bottom right shows, during manager training, a policy decision updates the bank, and a **frozen Answer Agent** reads the updated bank to produce an answer and score. That closes the loop from a memory edit to an answer reward.

### Slide 6: Appendix A: Memory Across Dialogue Turns (~1.2 min)

> **[Delivery Cue]** Trace the four columns from left to right, emphasizing the contrast between the Prompted Baseline and RL-Trained Memory-R1 at the bottom.

To see how all six operations connect across time, let's trace the two-adoption scenario from Appendix A. Note that this diagram simplifies a real dialogue trace to illustrate the mechanism rather than report an accuracy statistic.

- At the **First Adoption**, Andrew mentions adopting Buddy, and the manager executes `ADD` to store one adoption.  
- At the **Second Adoption**, Andrew mentions adopting *another* dog, Scout. After retrieving related memories, the word "another" is crucial. The prompted baseline on the bottom left treats Scout as a contradictory fact, deletes Buddy, and adds Scout. By contrast, RL-trained Memory-R1 executes `UPDATE`, merging Buddy and Scout into one consolidated record.  
- At **Question Time**, the question "How many dogs has Andrew adopted?" triggers `RETRIEVE`, returning up to 60 candidates.  
- At **Answer Time**, Memory-R1's `DISTILL` step selects the consolidated entry naming both Buddy and Scout and answers "2 dogs," whereas the baseline can only recover "1 dog" from its incomplete bank.

This illustrates why write-time errors are so costly: once an earlier fact is overwritten, even strong read-time retrieval cannot bring it back.

### Slide 7: 3.3 Constructing Training Examples (~1.4 min)

> **[Delivery Cue]** Explain why saving a memory snapshot is essential, and clearly distinguish the two separate data-construction paths for the Manager and the Answerer.

How does the paper construct training examples to capture these delayed effects? Slide 7 shows the three-stage pipeline, where Stage 2 and Stage 3 form **two separate data-construction paths**.

- **Stage 1: Rebuild the memory bank.** For any dialogue turn linked to a gold QA pair, the authors replay all prior sessions in temporal order using GPT-4o-mini to extract, merge, and save a **memory snapshot** representing the exact state of memory just before that turn. Why save a snapshot? Because it guarantees a controlled baseline: every candidate edit sampled by the manager is applied to the exact same prior bank and scored on the exact same downstream question by the frozen answerer.  
- **Stage 2: Train the write policy.** Starting from the snapshot, the manager sees the new fact and related prior memories, samples an edit, applies it, and receives a score from the frozen answerer.  
- **Stage 3: Train the evidence-selection policy.** Separately, for the Answer Agent, the system takes a question and retrieves up to 30 memory entries per speaker—up to 60 in total—and trains the agent to distill the candidates and generate the answer against the gold reference.

Across this setup, the "152" training count refers to 152 QA pairs from LoCoMo, paired with their full dialogue histories and reconstructed memory snapshots. As the bottom diagram reminds us, downstream answers provide the exact feedback signal needed to penalize harmful earlier edits.

### Slide 8: 3.4 PPO and GRPO for Memory Policies (~1.8 min)

> **[Delivery Cue]** Slow down slightly here. Explicitly clarify that the three-sample group at the top is an illustrative worked example derived from Equation 3, not a logged run. Walk step-by-step from rewards to standardized advantages, then contrast GRPO and PPO.

Now let's look at how PPO and GRPO optimize these policies. To make GRPO's group-relative advantage concrete, the top half presents a worked numerical example with group size $G = 3$ based on the dog-adoption scenario. To be clear, this is an illustrative example derived from Equation 3, not a reported training run.

In **01 Sample and Score**, from the same state $s$, the write policy samples three candidate edits:  
- Sample 1 executes `UPDATE` and merges both dogs, leading the frozen answerer to output "2 dogs" with Exact Match reward $r_1 = 1$.  
- Sample 2 overwrites Buddy with Scout, yielding "1 dog" and $r_2 = 0$.  
- Sample 3 chooses `NOOP`, keeping Buddy only, also yielding "1 dog" and $r_3 = 0$.

In **02 Compare Within a Group**, GRPO avoids training a separate critic network by standardizing rewards within the sampled group. For $r = (1, 0, 0)$, the mean is $\mu = 1/3$, and the population standard deviation is $\sigma = \sqrt{2}/3 \approx 0.471$. Applying $A_i = (r_i - \mu) / \sigma$ gives Sample 1 a positive advantage of $+1.41$, while Samples 2 and 3 each receive $-0.71$, with advantages summing to zero.

In **03 Interpret the Advantage**, this immediately tells the policy which sampled edit outperformed the group average for that exact input and should be reinforced.

Moving to **04 Update One Policy at a Time**:  
GRPO maximizes the expected importance-weighted advantage across the group, $\rho_i A_i$, minus a KL-divergence penalty $-\beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$ against a reference policy to prevent unstable drift.  
By contrast, **PPO** uses a learned critic network to estimate advantages and clips the importance ratio $\rho$ to $[1-\varepsilon, 1+\varepsilon]$.

On the bottom right, the training loop summarizes credit assignment: freeze the partner, sample from the active policy, score against the gold answer, and update—using the frozen answerer to score the manager, and exact match on its own output to score the Answer Agent.

### Slide 9: 3.5 Outcome-Based Reward Design (~1.1 min)

> **[Delivery Cue]** Clarify that the three rows are three separate examples whose scores are not summed, highlight the simplicity of binary EM with zero intermediate labels, and hand off to Presenter 3.

To wrap up the method section, Slide 9 summarizes the reward design. Both policies use a purely outcome-based **Binary Exact Match (EM)** reward: $r = 1$ if the generated answer $\hat{y}$ exactly matches the gold answer $y^*$, and $r = 0$ otherwise.

The three rows show three independent worked examples—their scores are separate and not summed:  
- Rows 01 and 02 show feedback to the **Memory Manager** through a frozen Answer Agent: preserving both dog names yields "2 dogs" and reward 1; overwriting Buddy yields "1 dog" and reward 0.  
- Row 03 shows the **Answer Agent's** own training reward with the manager fixed: given up to 60 retrieved candidates, distilling the right evidence and answering "2 dogs" earns reward 1.

Notice how sparse and label-free this supervision is: there are no ground-truth labels for which memory operation to pick or which candidate entry to select—everything is driven solely by final answer accuracy.

How well does this simple reward perform in practice, and where do its limits show up? I will now pass the floor to Presenter 3 for the experiments and discussion.

## Part 3: Experiments, Limitations & Discussion (Presenter 3 · ~7.5 min)

### Slide 10: 4.1 Experimental Setup & LoCoMo Overall Results (~1.4 min)

> **[Delivery Cue]** Briefly cover dataset splits, backbones, and the three metrics, then draw attention to the highlighted GRPO rows and the four Key Findings at the bottom.

Thank you. Let's now examine the experimental evidence and what it reveals.

Starting with the setup on top: the models are trained on **152 QA pairs** from LoCoMo, validated on 81 questions, and evaluated on **1,307 held-out LoCoMo questions** (excluding the adversarial category). The authors also test transfer to MSC and LongMemEval. Experiments use two open-weight instruction-tuned backbones: **LLaMA-3.1-8B-Instruct** and **Qwen-2.5-7B-Instruct**.

On the right are three 0–100 evaluation metrics: **F1** (token precision/recall harmonic mean), **B1** (unigram BLEU with brevity penalty), and **J** (an LLM-as-a-Judge semantic correctness score). Besides prompted memory baselines—LoCoMo RAG, A-Mem, Mem0, and MemoryOS—the table includes a strong supervised control, **Memory-SFT**, which imitates GPT-5 trajectories on the exact same training data.

Looking at the main table and the key findings at the bottom:  
First, **Memory-R1-GRPO leads on all six overall metrics** across both backbones, reaching **62.74** overall Judge score on LLaMA and **61.51** on Qwen—far ahead of all prompted memory systems.  
Second, compared to Memory-SFT, GRPO gains **+3.98 Judge points** on LLaMA and **+0.38 points** on Qwen.  
Third, notably, **PPO trails SFT on overall Judge score for both backbones** (57.54 vs. 58.76 on LLaMA; 59.53 vs. 61.13 on Qwen), showing that not every RL algorithm beats strong imitation under sparse rewards.

### Slide 11: 4.2 Main Results (~1.5 min)

> **[Delivery Cue]** Move left to right: Judge summary plots, lexical overlap bar charts, and then pause on the right-hand QA-type table to highlight the nuanced trade-off at the bottom.

Slide 11 unpacks the main results along three angles.

On the left, the Judge score plots show how Memory-R1-GRPO rises well above the mean of A-Mem, Mem0, MemoryOS, and Memory-SFT (49.35 on LLaMA and 51.62 on Qwen).

In the center, the lexical overlap charts show that compared to Memory-SFT, **GRPO consistently improves both F1 and BLEU-1 on both backbones**: on LLaMA, F1 rises by +2.21 and BLEU-1 by +4.53; on Qwen, F1 rises by +3.63 and BLEU-1 by +5.60. Under Exact Match training, answers become both more accurate and more concise.

However, the **QA-type breakdown on the right** reveals a much richer story:  
- On **LLaMA-3.1-8B**, GRPO's largest jump is on **temporal reasoning**, surging from 37.35 to 53.01 (**+15.66 points**), alongside solid gains on multi-hop and open-domain questions. Yet on **single-hop** questions, GRPO actually drops by 3.10 points relative to SFT (54.65 to 51.55).  
- On **Qwen-2.5-7B**, GRPO boosts multi-hop questions from 57.74 to 62.34, but drops on temporal questions to 40.96—where PPO actually leads Qwen with 53.01.

As the takeaway bar at the bottom emphasizes: **an overall average gain can coexist with regressions on specific question types**, depending on both the backbone and the RL algorithm.

### Slide 12: 4.3 Component Ablations (~1.2 min)

> **[Delivery Cue]** Clarify that all bars come from LLaMA-3.1-8B and each ablated variant is compared independently against Full R1 (not cumulative). Emphasize Manager RL's dominant impact and the PPO metric divergence.

Next, Slide 12 tests whether every component in Memory-R1 truly matters, using LLaMA-3.1-8B ablations. Each variant removes one component independently from the full system: untuned manager (`-Mgr RL`), untuned answerer (`-Ans RL`), or skipping evidence selection (`-Distill`). Each $\Delta$ equals the full system minus the ablated variant.

Looking at **GRPO** on the bottom:  
- Removing **Manager RL** causes the biggest drop in Judge score—**9.8 points** (62.7 down to 52.9)—confirming that learning *how to edit memory* is the primary driver of semantic accuracy.  
- Removing **Answerer RL** reduces Judge score by **2.8 points**, while dropping F1 by 12.0 points and BLEU-1 by 12.6 points.  
- Removing **Distillation** lowers Judge score from 62.7 to 60.1 (a 2.6-point drop), confirming that filtering noisy candidates before answering pays off.

Looking at **PPO** on top, notice one striking contrast: removing Answerer RL actually *raises* the PPO Judge score from 57.5 to 59.4 ($\Delta = -1.9$), even though F1 and BLEU-1 drop sharply. This is a sharp reminder that semantic judge scores and lexical overlap metrics measure different aspects of answer quality.

### Slide 13: 4.4 Policy and Reward Comparisons (~1.2 min)

> **[Delivery Cue]** Clarify that the curves plot Answerer training reward while the tables show full-system held-out test scores; then explain the Table 2 Judge-reward vs. Exact-Match trade-off at bottom right.

Slide 13 compares training dynamics and reward design.

The curves at the top (Figure 7 in the paper) track the **Answerer's** exact-match reward over 200 training steps, while the tables below report full-system held-out LoCoMo scores. Across both LLaMA and Qwen, **GRPO climbs much faster early in training**, whereas by step 200 both PPO and GRPO converge to a similar training reward around 0.30. Even so, GRPO generalizes better on the held-out test set for both models.

Now look at the bottom right—a crucial experiment on **reward choice** using LLaMA PPO (Table 2). One might ask: why not train directly with an LLM Judge reward instead of strict Exact Match? When the authors swap EM for Judge reward, the test Judge score indeed jumps from **57.54 to 63.58**, but **F1 falls from 41.05 to 33.69**, and **BLEU-1 drops from 32.91 to 23.36**. The model learns to produce longer, more verbose answers that satisfy the LLM judge at the expense of precision and brevity.

### Slide 14: Limitations, Evidence and Web Science Connections (~1.5 min)

> **[Delivery Cue]** Spend ~30 seconds on the two appendix tables (noting that the 7B scaling run is a separate experiment from Table 1), then spend ~1 minute mapping the three limitations to Web Science directions, noting that multi-probe evaluation under L1 is our group's discussion proposal.

Before concluding, Slide 14 adds two appendix results to test generalization and scaling.

On the top left is transfer to the **oracle version of LongMemEval**. Untuned base models score 44.2 Judge on LLaMA and 53.2 on Qwen. Memory-R1-GRPO transfers strongly, reaching 55.4 on LLaMA and 57.8 on Qwen (+4.6 over base). In contrast, Qwen PPO regresses by 5.8 Judge points below the untuned base (down to 47.4).

On the top right is the **Qwen scaling study** across 3B, 7B, and 14B models—note that this is a separate appendix experiment, so its 7B numbers differ from Table 1. Overall F1 improves with RL across all three scales; yet in the 7B scaling run, while GRPO raises F1 from 29.36 to 41.31, the Judge score slightly dips from 58.38 to 57.46—again showing metric tension.

Bringing everything back to **Web Science**, the bottom panel links three core limitations to future research directions:  
- **L1 — Answer-Level Supervision (Knowledge Maintenance):** A single QA probe only exposes part of an edit's effect. As a discussion proposal from our group, future evaluations should use **multi-probe testing** after each edit—checking both the updated fact and older facts that should still persist.  
- **L2 — Dialogue-Centric Evaluation (Knowledge Representation):** Current benchmarks cover multi-session text dialogues; expanding to multimodal and real-world web records remains open.  
- **L3 — Separately Trained Agents (Knowledge Integration):** Because one policy stays frozen while the other learns, studying stable **joint read–write training** is an important next step to see if coupled updates improve recall and utilization.

### Slide 15: Conclusion & Q&A (~0.7 min)

> **[Delivery Cue]** Deliver the closing synthesis calmly, pose the open discussion question to spark Q&A, and thank the audience.

To summarize: Memory-R1 demonstrates that downstream answer correctness alone can train two complementary RL policies—one that learns how to edit persistent memory, and one that learns how to distill evidence from noisy retrieval. At the same time, the experiments show that both components matter, and that their gains vary meaningfully across backbones, RL algorithms, and evaluation metrics.

Before we open the floor for questions, we would like to leave one thought-provoking question for discussion: **Going back to Andrew's two dogs—if a memory system accidentally forgets Buddy, but the next question happens to ask only about Scout, the system still gets full reward. What kind of evaluation protocol would actually detect that silent forgetting?**

Thank you very much for your attention! We are now happy to take your questions.

## Appendix: Quick Q&A Preparation Notes

1. **Why train the Memory Manager and Answer Agent separately instead of jointly end-to-end?**  
   Discrete memory edits and retrieval sit between the two policies. Updating both simultaneously makes the memory distribution and read policy shift at the same time, destabilizing credit assignment. Freezing the partner decouples the problem into two stationary RL tasks—while stable joint training remains an open future direction (Slide 14, L3).
2. **How can training on only 152 QA pairs generalize to 1,307 held-out LoCoMo questions and LongMemEval without severe overfitting?**  
   Although there are 152 QA pairs, each pair is backed by rich multi-session dialogue histories and reconstructed memory snapshots. Group sampling explores multiple candidate edits per state, and GRPO's KL penalty against the reference policy keeps the model from drifting—so the agents learn generalizable *meta-policies* for merging/preserving facts and filtering evidence rather than memorizing specific dialogues.
3. **Why is GRPO's Judge-score gain over SFT much larger on LLaMA (+3.98) than on Qwen (+0.38)?**  
   Qwen-2.5-7B starts from a higher Memory-SFT baseline (61.13 vs. 58.76 on LLaMA), and as Slide 11 shows, Qwen-GRPO gains strongly on multi-hop questions (57.74 → 62.34) but drops on temporal questions (46.99 → 40.96), narrowing the overall Judge margin—even though its F1 (+3.63) and BLEU-1 (+5.60) improvements remain substantial.
4. **Isn't Binary Exact Match too strict when a semantically correct answer is phrased differently?**  
   It is strict, which introduces a deliberate trade-off shown in Slide 13 (Table 2): replacing EM with an LLM Judge reward raises the LLaMA PPO Judge score from 57.54 to 63.58, but causes F1 to drop from 41.05 to 33.69 and BLEU-1 from 32.91 to 23.36 because the agent learns to generate verbose responses to please the judge. Binary EM keeps answers concise and precise.
