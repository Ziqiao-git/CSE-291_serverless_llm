class SpeculativeDecoder:
    def __init__(self, target_model_name: str, draft_model_name: str, device: str = "cuda"):
        """
        Initialize the speculative decoder with target and draft models.

        Args:
            target_model_name: HuggingFace model ID for the larger target model.
            draft_model_name: HuggingFace model ID for the smaller draft model.
            device: Device to run models on ("cuda" or "cpu").
        """
        self.device = device
        self.target_model, self.target_tokenizer = self.initialize_target_model(target_model_name)
        self.draft_model, self.draft_tokenizer = self.initialize_draft_model(draft_model_name)

        # Ensure tokenizers are compatible
        assert self.target_tokenizer.vocab == self.draft_tokenizer.vocab, "Tokenizers must be compatible"

    def initialize_target_model(self, model_name: str):
        """Initialize the larger target model with caching enabled and proper pad token."""
        print(f"Loading target model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # TODO: Implement target model initialization
        # 1. Set the pad token if it doesn't exist
        # 2. Load the model with appropriate settings for inference
        # 3. Enable any optimizations that might help with performance
                # 1. Set the pad token if missing (Llama might not have one).
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. Load the model in a reasonably precise mode, or float16 on GPU if you like.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        model.to(self.device)
        model.eval()

        # 3. Any additional performance optimizations or settings
        # For example: model.enable_input_require_grads(False), or half-precision, etc.
        # but for now let's keep it simple.

        return model, tokenizer


    def initialize_draft_model(self, model_name: str):
        """
        Initialize a smaller, faster draft model with proper pad token.
        Uses lower precision and additional optimizations.
        """
        print(f"Loading draft model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # TODO: Implement draft model initialization
        # 1. Set the pad token if it doesn't exist
        # 2. Load the model with appropriate settings for inference
        # 3. Enable any optimizations that might help with performance
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load the smaller "draft" model in half-precision for speed.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        model.to(self.device)
        model.eval()

        return model, tokenizer

    def generate_draft_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                             num_speculative_tokens: int = 10) -> torch.Tensor:
        """
        Generate speculative tokens in one forward call using the draft model.

        Args:
            input_ids: Input token IDs (tensor of shape [1, seq_len]).
            attention_mask: Corresponding attention mask.
            num_speculative_tokens: Number of tokens to speculate.

        Returns:
            Tensor of shape [1, num_speculative_tokens] containing the draft tokens.
        """
        # TODO: Implement draft token generation
        # 1. Use the draft model to generate tokens
        # 2. Extract only the new tokens (not including the input)
        # 3. Return the newly generated tokens

        with torch.no_grad():
            # In Transformers, we can directly limit the new tokens
            output_ids = self.draft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_speculative_tokens,
                do_sample=False,              # or True if you want sampling
                pad_token_id=self.draft_tokenizer.pad_token_id
            )
        # output_ids now has shape [1, seq_len + num_speculative_tokens]
        # We only want the newly generated portion:
        draft_tokens = output_ids[:, input_ids.shape[1]:]  # shape [1, num_speculative_tokens]
        return draft_tokens

    def verify_tokens_vectorized(self, input_ids: torch.Tensor, draft_tokens: torch.Tensor,
                               attention_mask: torch.Tensor) -> Tuple[List[int], int]:
        """
        Vectorized verification: verify all draft tokens in one forward pass using the target model.

        Args:
            input_ids: The current input token IDs (shape [1, L]).
            draft_tokens: Draft tokens from the draft model (shape [1, k]).
            attention_mask: The current attention mask for input_ids.

        Returns:
            accepted_tokens: List of accepted token IDs.
            accepted_position: Index of the first rejected token (if all accepted, equals draft_tokens.shape[1]).
        """
        # TODO: Implement efficient verification of draft tokens
        # 1. Run target model on input_ids concatenated with draft_tokens
        # 2. Extract the logits for positions where draft tokens would be predicted
        # 3. Compare target model predictions with draft tokens
        # 4. Determine how many consecutive tokens were accepted before first mismatch

        combined = torch.cat([input_ids, draft_tokens], dim=1)  # shape [1, L + k]
        new_mask = torch.ones_like(draft_tokens, dtype=attention_mask.dtype)
        combined_mask = torch.cat([attention_mask, new_mask], dim=1)

        # 2. Run one forward pass
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=combined,
                attention_mask=combined_mask,
                use_cache=False  # or True if you like
            )
        # logits shape: [1, L + k, vocab_size]
        logits = outputs.logits

        L = input_ids.shape[1]
        k = draft_tokens.shape[1]
        accepted_tokens = []
        mismatch_idx = k  # if we never mismatch, it stays k

        # 3. For the i-th new token, check target's top-1 at position (L + i - 1)
        #    Because the token at index (L + i - 1) predicts the token at (L + i).
        for i in range(k):
            # The distribution is in logits[0, L + i - 1, :] (the next token predicted after that position)
            pos = (L + i) - 1  # e.g. if i=0 => pos=L-1
            if pos < 0:
                # corner case if L=0 and i=0?
                continue

            next_token_dist = logits[0, pos, :]  # shape [vocab_size]
            pred_id = torch.argmax(next_token_dist)
            if pred_id == draft_tokens[0, i]:
                accepted_tokens.append(int(draft_tokens[0, i].item()))
            else:
                mismatch_idx = i
                break

        return accepted_tokens, mismatch_idx

    def speculative_decode(self, prompt: str, max_tokens: int = 100,
                          num_speculative_tokens: int = 15) -> str:
        """
        Main speculative decoding algorithm with vectorized verification.

        Args:
            prompt: Input text.
            max_tokens: Maximum number of tokens to generate (excluding prompt).
            num_speculative_tokens: Number of tokens to speculate per iteration.

        Returns:
            Generated text.
        """
        # Tokenize prompt
        inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_length = input_ids.shape[1]

        # Initialize counters for performance tracking
        total_tokens_generated = prompt_length
        total_draft_tokens_proposed = 0
        total_draft_tokens_accepted = 0
        start_time = time.time()

        # TODO: Implement the core speculative decoding loop
        # 1. Generate draft tokens using the draft model
        # 2. Verify draft tokens using the target model
        # 3. Accept verified tokens and append to the sequence
        # 4. For rejected tokens or if all tokens are accepted, generate a new token with the target model
        # 5. Stop when max_tokens is reached or an EOS token is generated
                # 2. Loop until max_tokens is reached or we hit an EOS
        while (total_tokens_generated - prompt_length) < max_tokens:
            # Generate draft tokens
            draft_tokens = self.generate_draft_tokens(input_ids, attention_mask,
                                                      num_speculative_tokens=num_speculative_tokens)
            k = draft_tokens.shape[1]
            total_draft_tokens_proposed += k

            # Verify them with the target
            accepted, mismatch_idx = self.verify_tokens_vectorized(input_ids, draft_tokens, attention_mask)

            # Accept the tokens that matched
            if len(accepted) > 0:
                accepted_tensor = torch.tensor([accepted], dtype=torch.long, device=self.device)
                input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
                mask_new = torch.ones_like(accepted_tensor)
                attention_mask = torch.cat([attention_mask, mask_new], dim=1)

                total_tokens_generated += len(accepted)
                total_draft_tokens_accepted += len(accepted)

            # If mismatch happened before we accepted them all, generate exactly 1 from target
            if mismatch_idx < k:
                # We'll do one normal decode step for the mismatch
                with torch.no_grad():
                    output_ids = self.target_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=self.target_tokenizer.pad_token_id
                    )
                new_tok = output_ids[:, input_ids.shape[1]:]
                input_ids = torch.cat([input_ids, new_tok], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(new_tok)], dim=1)
                total_tokens_generated += 1

            # Check if we hit EOS
            last_token = input_ids[0, -1].item()
            if last_token == self.target_tokenizer.eos_token_id:
                break

            if (total_tokens_generated - prompt_length) >= max_tokens:
                break

        # Calculate performance metrics
        elapsed_time = time.time() - start_time
        acceptance_rate = total_draft_tokens_accepted / total_draft_tokens_proposed if total_draft_tokens_proposed > 0 else 0

        print(f"Generated {total_tokens_generated - prompt_length} tokens in {elapsed_time:.2f} seconds")
        print(f"Tokens per second: {(total_tokens_generated - prompt_length) / elapsed_time:.2f}")
        print(f"Draft token acceptance rate: {acceptance_rate:.2%}")

        return self.target_tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def benchmark(self, prompt: str, max_tokens: int = 100,
                  num_runs: int = 3, compare_baseline: bool = True) -> Dict:
        """
        Benchmark the speculative decoder against baseline decoding.

        Args:
            prompt: Input text.
            max_tokens: Maximum number of tokens to generate.
            num_runs: Number of benchmark runs.
            compare_baseline: Whether to compare with baseline (non-speculative) decoding.

        Returns:
            Dictionary with benchmark results.
        """
        results = {
            "speculative": {"times": [], "tokens_per_second": []},
            "baseline": {"times": [], "tokens_per_second": []} if compare_baseline else None
        }

        # Benchmark speculative decoding.
        for _ in range(num_runs):
            start_time = time.time()
            output = self.speculative_decode(prompt, max_tokens=max_tokens)
            elapsed = time.time() - start_time
            prompt_len = len(self.target_tokenizer(prompt)["input_ids"])
            output_tokens = len(self.target_tokenizer.encode(output)) - prompt_len
            tps = output_tokens / elapsed
            results["speculative"]["times"].append(elapsed)
            results["speculative"]["tokens_per_second"].append(tps)

        # Benchmark baseline decoding.
        if compare_baseline:
            for _ in range(num_runs):
                inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                start_time = time.time()
                with torch.no_grad():
                    output_ids = self.target_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + max_tokens,
                        do_sample=False,
                        pad_token_id=self.target_tokenizer.pad_token_id
                    )
                elapsed = time.time() - start_time
                output_tokens = output_ids.shape[1] - input_ids.shape[1]
                tps = output_tokens / elapsed
                results["baseline"]["times"].append(elapsed)
                results["baseline"]["tokens_per_second"].append(tps)

        for method in results.keys():
            if results[method] is not None:
                avg_time = sum(results[method]["times"]) / num_runs
                avg_tps = sum(results[method]["tokens_per_second"]) / num_runs
                results[method]["avg_time"] = avg_time
                results[method]["avg_tokens_per_second"] = avg_tps

        if compare_baseline:
            speedup = results["baseline"]["avg_time"] / results["speculative"]["avg_time"]
            results["speedup"] = speedup
            results["latency_reduction"] = (1 - results["speculative"]["avg_time"] / results["baseline"]["avg_time"]) * 100
            # print(f"Speculative decoding speedup: {speedup:.2f}x")
            # print(f"Latency reduction: {results['latency_reduction']:.2f}%")

        return results
    


# Initialize speculative decoder
decoder = SpeculativeDecoder(
    target_model_name=target_model_name,
    draft_model_name=draft_model_name,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Test prompts
test_prompts = [
    "The future of Artificial Intelligence is",
    "Write a short story about a robot learning to feel emotions:",
    "Write the lyrics to the song 'Happy Birthday'."
]

# Run benchmark on test prompts
for i, prompt in enumerate(test_prompts):
    print(f"\nBenchmarking Prompt {i+1}:")
    print(f"Prompt: {prompt}")

    results = decoder.benchmark(
        prompt=prompt,
        max_tokens=100,
        num_runs=3,
        compare_baseline=True
    )

    print(f"Average speculative decoding time: {results['speculative']['avg_time']:.2f} seconds")
    print(f"Average speculative tokens per second: {results['speculative']['avg_tokens_per_second']:.2f}")

    if results["baseline"] is not None:
        print(f"Average baseline decoding time: {results['baseline']['avg_time']:.2f} seconds")
        print(f"Average baseline tokens per second: {results['baseline']['avg_tokens_per_second']:.2f}")
        print(f"Speedup: {results['speedup']:.2f}x")
        print(f"Latency reduction: {results['latency_reduction']:.2f}%")