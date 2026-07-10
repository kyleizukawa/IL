"""
Rollout collector for GRPO-based Intuition Learning (v2).

Two-stage generation approach (aligned with the benchmark run_full.py):
1. Let the model think freely for up to `thinking_tokens` (exploration).
   - If it emits </think> on its own, great — proceed to answer generation.
   - If it hits the thinking budget without </think>, force </think> to
     transition to the answer phase (standard for bounded-compute reasoning).
2. Let the model generate the answer (grid) for up to `prediction_tokens`.
3. Parse the grid, compute accuracy, give feedback, repeat.

ALL generated tokens (thinking + forced </think> + answer) are actions that
receive GRPO gradients. The forced </think> token is a single token — its
logprob is included in the action logprobs, so the policy can learn to emit
it earlier on its own.

The thinking IS the exploration. The prediction IS the commitment.
IL rewards the quality of the understanding path.

Records:
- Full episode token sequence
- Action positions (all model-generated tokens, including forced </think>)
- Old logprobs (for GRPO ratio + KL computation)
- Total episode reward
"""
import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from .env import RLEnvironment

# DeepSeek-R1-Distill token IDs
THINK_OPEN_TOKEN = 151648   # <think>
THINK_CLOSE_TOKEN = 151649  # </think>


def collect_rollout(
    model,
    tokenizer,
    puzzle,
    n_steps=2,
    gamma=0.9,
    thinking_tokens=256,
    prediction_tokens=128,
    temperature=0.8,
    top_p=0.9,
    seed=None,
    **_kwargs,
):
    """Collect a single multi-step rollout with two-stage generation.

    Stage 1: free thinking (up to thinking_tokens). If the model emits
    </think> naturally, stop thinking early. If it hits the budget, force
    </think> to transition to the answer.
    Stage 2: answer generation (up to prediction_tokens). The model outputs
    its prediction (grid).

    Returns a dict with:
        tokens: list of all token IDs in the episode
        action_positions: list of (start, end) ranges for model-generated tokens
        old_logprobs: list of logprobs for each action token
        reward: total episode reward
        env: the RLEnvironment instance
        messages: the chat messages for multi-step interaction
        n_steps: number of steps completed
    """
    env = RLEnvironment(puzzle, n_steps=n_steps, gamma=gamma)

    all_tokens = []
    action_positions = []
    old_logprobs = []
    msgs = []

    for step in range(n_steps):
        # ── Build prompt for this step ──
        if step == 0:
            user_msg = env.build_initial_prompt()
            msgs = [{'role': 'user', 'content': user_msg}]
        else:
            feedback_text = env.build_feedback_prompt(step, prev_accuracy, prev_grid)
            msgs.append({'role': 'user', 'content': feedback_text})

        # Apply chat template to get the full token sequence
        full_prompt = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        )
        # Only add the NEW tokens (everything after what we already have)
        new_tokens = full_prompt[len(all_tokens):]
        all_tokens.extend(new_tokens)

        # ── Stage 1: Thinking (free generation up to thinking_tokens) ──
        if seed is not None:
            mx.random.seed(seed + step * 1000)
        sampler = make_sampler(temp=temperature, top_p=top_p)

        gen_start = len(all_tokens)
        gen_ids = []
        gen_logprobs = []
        think_done = False

        prompt_arr = mx.array(list(all_tokens))
        for token_id, logprobs in generate_step(
            prompt_arr, model, max_tokens=thinking_tokens, sampler=sampler
        ):
            gen_ids.append(token_id)
            gen_logprobs.append(float(logprobs[token_id]))
            all_tokens.append(token_id)
            # Stop thinking if the model emits </think> or EOS
            if token_id == THINK_CLOSE_TOKEN:
                think_done = True
                break
            if token_id == tokenizer.eos_token_id:
                think_done = True
                break

        # If the model didn't emit </think> naturally, force it.
        # This is standard for bounded-compute reasoning (see run_full.py).
        # The forced token IS an action (gets a logprob) so the policy can
        # learn to emit it earlier.
        if not think_done:
            gen_ids.append(THINK_CLOSE_TOKEN)
            gen_logprobs.append(0.0)  # placeholder logprob for forced token
            all_tokens.append(THINK_CLOSE_TOKEN)
            # Force a newline after </think> for clean formatting
            gen_ids.append(198)  # \n
            gen_logprobs.append(0.0)
            all_tokens.append(198)

        # ── Stage 2: Answer generation (up to prediction_tokens) ──
        if seed is not None:
            mx.random.seed(seed + step * 1000 + 1)
        sampler = make_sampler(temp=temperature, top_p=top_p)

        prompt_arr = mx.array(list(all_tokens))
        for token_id, logprobs in generate_step(
            prompt_arr, model, max_tokens=prediction_tokens, sampler=sampler
        ):
            gen_ids.append(token_id)
            gen_logprobs.append(float(logprobs[token_id]))
            all_tokens.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break

        gen_end = len(all_tokens)
        if gen_end > gen_start:
            action_positions.append((gen_start, gen_end))
            old_logprobs.extend(gen_logprobs)

        # ── Process the action ──
        # The full response is everything the model generated this step
        # (thinking + forced </think> + answer). parse_grid finds 2D arrays
        # anywhere in the text.
        generated_text = tokenizer.decode(gen_ids)
        reward, accuracy, grid, done = env.process_action(generated_text)

        # Add assistant message for next step's prompt
        msgs.append({'role': 'assistant', 'content': generated_text})

        prev_accuracy = accuracy
        prev_grid = grid

        if done:
            break

    return {
        'tokens': all_tokens,
        'action_positions': action_positions,
        'old_logprobs': old_logprobs,
        'reward': env.total_reward(),
        'env': env,
        'messages': msgs,
        'n_steps': len(action_positions),
    }
