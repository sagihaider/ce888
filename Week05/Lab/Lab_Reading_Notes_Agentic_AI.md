# Reading Notes: Introduction to AI Agents and Agentic AI
## CE888 - Data Science and Decision-Making
## Background Material for Lab Sessions

---

## PURPOSE OF THESE NOTES

These notes provide essential background knowledge about AI agents and agentic AI systems. Read these **before** starting the lab exercises to understand the context and motivation behind the coding questions you'll be working on.

**Estimated reading time:** 30-40 minutes  
**Recommended:** Read through once, then refer back while doing lab exercises

---

## PART 1: WHAT IS AN AI AGENT?

### Basic Definition

An **AI agent** is a system that:
1. Perceives its environment (gathers information)
2. Makes decisions (reasons about what to do)
3. Takes actions (executes its decisions)
4. Works toward achieving specific goals

**Key difference from traditional programs:** Agents have **autonomy**—they can decide what to do based on the situation, rather than just following a fixed script.

### Real-World Analogy

Think of a human data analyst:
- **Perceives:** Looks at a dataset, notices missing values
- **Reasons:** "I need to handle these missing values before analyzing"
- **Acts:** Chooses an imputation method and applies it
- **Adapts:** If one approach doesn't work, tries another

An AI agent does the same thing, but autonomously.

### The Classical Agent Loop

Every agent operates in a continuous cycle:

```
┌─────────────┐
│  PERCEIVE   │ ← Gather information about current state
└──────┬──────┘
       ↓
┌──────────────┐
│   REASON     │ ← Decide what action to take
└──────┬───────┘
       ↓
┌──────────────┐
│     ACT      │ ← Execute the chosen action
└──────┬───────┘
       ↓
    [Observe results]
       ↓
    [Loop back to PERCEIVE]
```

This loop continues until the agent achieves its goal or determines it cannot succeed.

---

## PART 2: TRADITIONAL ML VS. AGENTIC AI

### Traditional Machine Learning Approach

**How it works:**
```
Input Data → [Trained Model] → Output Prediction
```

**Characteristics:**
- **One-way flow:** Input goes in, output comes out
- **Single decision:** Makes one prediction and stops
- **No reasoning:** Cannot explain its thought process
- **No tools:** Limited to what's in its parameters
- **Fixed:** Cannot adapt during inference

**Example:**
```python
# Traditional ML
prediction = model.predict(data)
# That's it - just one prediction
```

### Agentic AI Approach

**How it works:**
```
Input Goal → [Agent Loop: Perceive → Reason → Act] → Goal Achieved
```

**Characteristics:**
- **Interactive:** Can take multiple actions
- **Reasoning:** Thinks through each step
- **Tool use:** Can call functions, APIs, databases
- **Adaptive:** Changes approach if something fails
- **Iterative:** Keeps working until task complete

**Example:**
```python
# Agentic AI
agent.goal = "Analyze this dataset"

# Agent thinks: "I'll check data quality first"
result = agent.execute_tool("check_missing_values", data)

# Agent observes: "There are missing values"
# Agent thinks: "I need to handle these"
agent.execute_tool("impute_missing", data, method="median")

# Agent thinks: "Now I can analyze"
agent.execute_tool("calculate_statistics", data)

# Agent: "Task complete!"
```

### Key Differences Summary

| Aspect | Traditional ML | Agentic AI |
|--------|----------------|------------|
| **Execution** | Single pass | Multiple steps |
| **Reasoning** | Hidden | Explicit |
| **Tools** | None | Can use many |
| **Adaptation** | Cannot adapt | Can change approach |
| **Autonomy** | Low | High |

---

## PART 3: WHY AGENTIC AI MATTERS NOW

### The Evolution of AI Systems

**1960s-1980s: Rule-Based Agents**
- Hand-coded rules: "IF this THEN that"
- Deterministic and predictable
- Limited to programmer's knowledge
- Example: Expert systems for medical diagnosis

**2000s-2015: Machine Learning Systems**
- Learn patterns from data
- Can generalize to new examples
- But: No reasoning, just predictions
- Example: Image classifiers, recommendation systems

**2020s-Present: LLM-Based Agents**
- Can understand natural language instructions
- Can reason through complex problems
- Can use tools and APIs
- Can explain their thinking
- Example: ChatGPT with plugins, Claude with tools

### Why Now? Three Key Enablers

**1. Large Language Models (LLMs)**
- Can understand complex instructions in natural language
- Can generate reasoning traces
- Can decide which tools to use
- Provide the "brain" for the agent

**2. Function Calling / Tool Use**
- LLMs can now reliably call external functions
- Bridges language understanding to code execution
- Enables grounding in real data and computations

**3. Prompt Engineering Advances**
- Techniques like Chain-of-Thought and ReAct
- Structured ways to make LLMs reason step-by-step
- Methods for self-correction and reflection

**The Magic Combination:**
```
LLM Reasoning + Tool Use + Structured Prompting = Autonomous Agents
```

---

## PART 4: CORE CONCEPTS FOR THE LAB

### Concept 1: State Tracking

Agents need to remember what they've done and what they know.

**Example from lab:**
```python
# The agent tracks its progress
completed_tasks = ["load_data", "clean_data"]
all_tasks = ["load_data", "clean_data", "analyze", "report"]

# Agent knows: 2 done, 2 remaining
```

**Why it matters:**
- Prevents repeating work
- Enables planning next steps
- Allows progress monitoring

### Concept 2: Decision Making

Agents make choices based on conditions.

**Example from lab:**
```python
# Agent decides which statistical test to use
if sample_size < 30 and data_is_normal:
    use_t_test()
elif sample_size < 30 and not data_is_normal:
    use_non_parametric_test()
else:
    use_z_test()
```

**Why it matters:**
- Different situations require different actions
- Agents must choose the right tool for the job
- Decisions affect success or failure

### Concept 3: Tool Use

Agents accomplish tasks by calling functions (tools).

**Example from lab:**
```python
# Agent has a toolbox
available_tools = ["calculator", "database", "plotter"]

# Agent checks what tools it has
if "calculator" in available_tools:
    result = use_tool("calculator", "2 + 2")
```

**Why it matters:**
- Agents aren't limited to their internal knowledge
- Can perform precise computations
- Can access external data sources

### Concept 4: Retry Logic

Agents don't give up after one failure.

**Example from lab:**
```python
# Agent tries multiple times
attempts = 0
success = False

while attempts < max_retries and not success:
    result = try_action()
    if result == True:
        success = True
    attempts += 1
```

**Why it matters:**
- Tools can fail (network errors, timeouts)
- Agents need resilience
- But shouldn't retry forever (waste resources)

### Concept 5: Confidence and Uncertainty

Agents track how confident they are in their outputs.

**Example from lab:**
```python
# Agent tracks prediction accuracy
correct_predictions = 8
total_predictions = 10
confidence = correct_predictions / total_predictions  # 0.8 = 80%

if confidence > 0.7:
    status = "high confidence - trust this output"
else:
    status = "low confidence - needs human review"
```

**Why it matters:**
- Users need to know when to trust the agent
- Low confidence signals need for human oversight
- Helps agents improve over time

### Concept 6: Task Prioritization

Agents organize work by importance.

**Example from lab:**
```python
# Agent sorts tasks by priority
tasks = [
    {"name": "fix_critical_bug", "priority": "high"},
    {"name": "write_docs", "priority": "low"},
    {"name": "review_code", "priority": "medium"}
]

# Agent works on high priority first
ordered_tasks = ["fix_critical_bug", "review_code", "write_docs"]
```

**Why it matters:**
- Not all tasks are equally important
- Resources (time, compute) are limited
- Prioritization maximizes impact

---

## PART 5: THE REACT PATTERN (CORE FRAMEWORK)

### What is ReAct?

**ReAct** = **Rea**soning + **Act**ing

It's a pattern where the agent alternates between:
1. **Thought** - Reasoning about what to do
2. **Action** - Using a tool or taking an action  
3. **Observation** - Seeing what happened

### ReAct Example

**Task:** "Analyze correlation between age and income in dataset"

**Agent's ReAct Loop:**

```
THOUGHT: I need to first check if the data is loaded and valid
ACTION: check_data_loaded()
OBSERVATION: Data is loaded, 1000 rows, 5 columns

THOUGHT: Before calculating correlation, I should check for missing values
ACTION: check_missing_values(columns=["age", "income"])
OBSERVATION: age has 3% missing, income has 5% missing

THOUGHT: Small amount of missing data, I'll remove those rows
ACTION: remove_missing_rows(columns=["age", "income"])
OBSERVATION: 920 rows remaining after removing missing values

THOUGHT: Now I can calculate the correlation
ACTION: calculate_correlation(col1="age", col2="income")
OBSERVATION: Correlation = 0.73, p-value = 0.001

THOUGHT: Strong positive correlation, statistically significant. Task complete!
```

### Why ReAct Works

**1. Transparency:** You can see the agent's reasoning
**2. Debuggability:** If something goes wrong, you know where
**3. Correctness:** Agent checks results before proceeding
**4. Adaptability:** Agent can change approach based on observations

### ReAct in Your Lab Code

Many lab questions implement parts of the ReAct pattern:

- **State tracking** → Keeping track of what's been done
- **Decision making** → The "Thought" step
- **Tool execution** → The "Action" step  
- **Result checking** → The "Observation" step

---

## PART 6: REFLECTION AND SELF-IMPROVEMENT

### What is Reflection?

**Reflection** is when an agent evaluates its own output and improves it.

**The Reflection Loop:**
```
1. GENERATE: Create initial solution
2. EVALUATE: Critique the solution
3. DECIDE: Good enough? If yes, stop. If no, continue.
4. REFINE: Improve the solution based on critique
5. REPEAT: Go back to step 2
```

### Reflection Example

**Task:** "Write a function to calculate average"

**Iteration 1:**
```python
# Initial solution
def average(numbers):
    return sum(numbers) / len(numbers)
```

**Self-Critique:**
"This will crash if the list is empty! I need to handle that edge case."

**Iteration 2:**
```python
# Improved solution
def average(numbers):
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)
```

**Self-Critique:**
"Good! Now it handles empty lists. But wait - is returning 0 the right behavior? Maybe I should return None or raise an error?"

**Iteration 3:**
```python
# Final solution
def average(numbers):
    if len(numbers) == 0:
        return None  # Clearer than 0
    return sum(numbers) / len(numbers)
```

**Self-Critique:**
"Perfect! Handles edge cases correctly and returns None for empty input, which is clearer than 0."

### Why Reflection Matters

**Without Reflection:**
- First attempt might have bugs
- No quality checking
- Errors go unnoticed

**With Reflection:**
- Agent catches its own mistakes
- Improves solution iteratively
- Higher quality final output

**Trade-off:**
- More iterations = better quality
- But: More time and computational cost
- Need to balance quality vs. efficiency

### Reflection in Lab

The confidence tracking question (Question 3 in medium level) demonstrates a form of reflection:
- Agent tracks success/failure of predictions
- Adjusts confidence based on performance
- Learns from feedback over time

---

## PART 7: PRACTICAL IMPLICATIONS FOR YOUR LAB WORK

### What You're Actually Building

In the lab exercises, you're building the **fundamental building blocks** of an agentic system:

**Simple Questions (Easy):**
- Counting actions → State awareness
- Checking completion → Progress tracking
- Status checking → Agent states
- Finding next task → Planning
- Success rates → Performance monitoring
- Tool availability → Resource checking

**Medium Questions:**
- Retry logic → Resilience and error handling
- Task scheduling → Prioritization and planning
- Confidence tracking → Self-assessment and learning

**These aren't toy exercises!** These are the actual mechanisms that production agent systems use.

### How Lab Concepts Map to Real Agents

**Lab Question** → **Real Agent Capability**

1. **Count actions** → Agents track their action history for logging and debugging

2. **Check if done** → Agents need to know when they've achieved their goal

3. **Get next task** → Agents plan their work queue

4. **Retry logic** → Real agents handle API failures, network timeouts, tool errors

5. **Task scheduler** → Production agents prioritize critical tasks over routine ones

6. **Confidence tracker** → Real agents need to know when their outputs are reliable

### Connecting to Your Coursework

Your coursework asks you to build an **Offline Agentic Data Scientist**. This means building a system that can:

1. **Perceive:** Load and inspect datasets
2. **Reason:** Decide which analyses to run
3. **Act:** Execute statistical tools
4. **Reflect:** Check if analysis is complete and correct
5. **Adapt:** Handle errors and edge cases

**The lab exercises teach you HOW to build each piece!**

---

## PART 8: COMMON MISCONCEPTIONS

### Misconception 1: "Agents are just fancy chatbots"

**Reality:** Chatbots generate text responses. Agents take actions and use tools.

- Chatbot: "You should probably check for missing values"
- Agent: *Actually runs check_missing_values() and shows you the results*

### Misconception 2: "Agents are just running if-else statements"

**Reality:** While agents use conditionals, they decide which conditions to check and what actions to take based on natural language understanding and reasoning.

- Traditional program: Follow fixed flowchart
- Agent: Understand goal, plan approach, adapt to unexpected situations

### Misconception 3: "More autonomy is always better"

**Reality:** Autonomy is a spectrum. You want the right level for your use case.

- **Low autonomy:** Agent suggests, human decides (good for critical decisions)
- **Medium autonomy:** Agent acts, human reviews (good for most tasks)
- **High autonomy:** Agent acts independently (good for routine tasks)

### Misconception 4: "Agents always make the right decision"

**Reality:** Agents can make mistakes! That's why we need:
- Confidence scores (to know when to trust output)
- Logging (to debug what went wrong)
- Human oversight (for critical decisions)
- Retry logic (to handle failures)

### Misconception 5: "I need to understand LLMs deeply to build agents"

**Reality:** You need to understand agent design patterns and system architecture. The LLM is just one component.

Focus on:
- How to structure agent loops
- When to use which tools
- How to handle errors
- How to evaluate quality

Not on:
- Transformer architecture details
- Training algorithms
- Internal representations

---

## PART 9: KEY TAKEAWAYS FOR LAB SUCCESS

### Essential Concepts (Master These!)

1. **State Management**
   - Keep track of what's been done
   - Store relevant information
   - Use variables to track progress

2. **Conditional Logic**
   - Make decisions based on conditions
   - Choose different actions for different situations
   - Handle edge cases

3. **Iteration**
   - Use loops for repeated actions
   - Know when to stop (termination conditions)
   - Track progress through iterations

4. **Data Structures**
   - Lists for ordered sequences
   - Dictionaries for structured information
   - Choosing the right structure for the task

5. **Error Handling**
   - Things can and will go wrong
   - Build in retry mechanisms
   - Provide informative error messages

### Practical Tips

**Before starting each lab question:**
1. Read the problem description carefully
2. Identify which agent concept it teaches
3. Think about how this would work in a real agent
4. Write pseudocode before actual code
5. Test with the provided examples first

**While coding:**
1. Start simple - get basic version working first
2. Add complexity incrementally
3. Test after each addition
4. Use print statements to debug
5. Check edge cases (empty lists, zero values, None)

**After completing:**
1. Review the solution provided
2. Compare your approach to alternatives
3. Think about how to extend it
4. Consider what could go wrong in production

### Connecting Lab to Lecture

The lab exercises implement concepts from specific lecture sections:

**Lab Questions** → **Lecture Sections**

- State tracking → Section 4: Practical Tips (State Management)
- Decision making → Section 1: Introduction (Agent Architecture)
- Tool use → Section 3: Tool Use
- Retry logic → Section 4: Practical Tips (Error Handling)
- Prioritization → Section 5: Planning Agents
- Confidence → Section 7: Critical Thinking (Reliability)

**Pro tip:** Refer back to lecture slides when stuck on lab questions!

---

## PART 10: PREPARING FOR THE GRADED LAB

### What to Expect

The graded lab will have questions similar to the practice questions, but:
- May combine multiple concepts
- May have less explicit guidance
- Will require you to think through the logic yourself
- Will test edge cases more thoroughly

### How to Prepare

**1. Complete ALL practice questions**
- Don't skip the easy ones
- They build foundational skills
- Try solving without looking at solutions first

**2. Understand, don't memorize**
- Don't just copy solutions
- Understand WHY each line works
- Be able to explain your code

**3. Practice explaining your code**
- Write comments explaining each section
- Could you teach it to someone else?
- Can you modify it for a different scenario?

**4. Test extensively**
- Try inputs not in the examples
- What happens with empty lists?
- What if numbers are negative?
- What if strings contain special characters?

**5. Review common patterns**
- Looping through lists
- Checking conditions
- Building dictionaries
- Counting and tracking

### Common Mistakes to Avoid

❌ **Not reading the question carefully**
- Missing important requirements
- Misunderstanding what to return
- Ignoring edge cases

❌ **Not testing with examples**
- Code that looks right but fails tests
- Off-by-one errors
- Wrong data type returned

❌ **Overcomplicating solutions**
- Simple is usually better
- Don't add unnecessary complexity
- Follow the hints provided

❌ **Not handling edge cases**
- Empty lists
- Zero values
- None values
- Single-item lists

❌ **Syntax errors**
- Missing colons after if/for
- Incorrect indentation
- Using = instead of ==
- Forgetting return statements

### Success Checklist

Before submitting any lab solution, verify:

✓ Function name matches exactly  
✓ Returns correct data type (list, dict, int, bool, etc.)  
✓ Handles all edge cases mentioned  
✓ Passes all provided test cases  
✓ Code is readable with clear variable names  
✓ No syntax errors  
✓ Logic is correct, not just lucky with test cases  

---

## PART 11: FURTHER READING (OPTIONAL)

If you want to explore these concepts more deeply:

**Easy Introductions:**
- Lilian Weng's blog: "LLM Powered Autonomous Agents"
- Anthropic documentation on Claude tool use
- LangChain documentation on agents

**Academic Papers:**
- ReAct paper (Yao et al., 2023)
- Reflexion paper (Shinn et al., 2023)
- Agent survey paper (Wang et al., 2024)

**Practical Resources:**
- LangChain tutorials
- OpenAI function calling examples
- AutoGPT project (on GitHub)

See the "Top 10 Essential Resources" document for full links and details.

---

## SUMMARY: THE BIG PICTURE

### What You've Learned

**Agentic AI** is about building systems that can:
1. Understand goals in natural language
2. Break complex tasks into steps
3. Use tools to accomplish each step
4. Adapt when things go wrong
5. Reflect on and improve their outputs
6. Work autonomously toward objectives

### Why It Matters

**Traditional AI:**
- "Here's a prediction"

**Agentic AI:**
- "I understand your goal. Let me handle this entire task for you. I'll figure out what to do, use the tools I need, check my work, and let you know when it's done."

**This is transformative because:**
- Moves from prediction to action
- Reduces human involvement in routine tasks
- Handles complex multi-step workflows
- Can work 24/7 without fatigue
- Scales to thousands of tasks

### Your Role

In this course, you're learning to **design and build** these systems. The lab exercises teach you the fundamental patterns. Your coursework applies them to a real problem. Your exam tests your understanding of when and how to use each pattern.

**You're learning skills that are in high demand right now!**

---

## FINAL WORDS OF ENCOURAGEMENT

Building agents is challenging because:
- It requires understanding multiple concepts
- You need to think through logic step-by-step
- Edge cases can be tricky
- Debugging requires patience

**But it's also incredibly rewarding:**
- You're building systems that actually DO things
- The concepts apply to real-world problems
- Every bug you fix teaches you something
- By the end, you'll have built an autonomous system!

**Remember:**
- Start with the simple questions
- Build up gradually
- Don't hesitate to ask for help
- Learn from mistakes
- Practice, practice, practice

**You've got this! Good luck with the lab! 🚀**

---

## QUICK REFERENCE: KEY TERMS

**Agent:** A system that perceives, reasons, and acts toward goals  
**Autonomy:** Ability to make decisions independently  
**Tool:** A function the agent can call to perform actions  
**State:** Information the agent tracks about its progress  
**ReAct:** Pattern of Reasoning → Acting → Observing  
**Reflection:** Agent evaluating and improving its own outputs  
**Retry Logic:** Trying multiple times when actions fail  
**Confidence:** How certain the agent is about its outputs  
**Planning:** Deciding sequence of actions to achieve goals  
**Termination:** Knowing when to stop trying  

---

**Version:** 1.0  
**Last Updated:** February 2026  
**For:** MSc AI Lab Sessions on Agentic AI
