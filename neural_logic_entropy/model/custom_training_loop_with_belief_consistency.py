import time
import numpy as np
import tensorflow as tf

def train_with_full_weight_logging_belief(
        model,
        input_data,
        output_data,
        sample_weight,
        optimizer,
        memmap_path_weights,
        memmap_path_inputs,
        memmap_path_outputs,
        val_split=0.1,
        epochs=1,
        batch_size=16,
        print_every=10000,
        early_stop_patience=10,
        random_seed=None,
        lambda_belief=1e-4
):
    """
    Custom training loop with:
      - mini-batch training for belief consistency
      - per-sample memmap logging
      - early stopping
      - small belief consistency nudge on hidden layer
    """

    num_samples = input_data.shape[0]
    input_size = input_data.shape[1]
    output_size = output_data.shape[1]
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val

    # --- Set random seed ---
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    # --- Random train-validation split ---
    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train_inputs = input_data[train_idx]
    train_outputs = output_data[train_idx]
    train_mask = sample_weight[train_idx]

    val_inputs = input_data[val_idx]
    val_outputs = output_data[val_idx]
    val_mask = sample_weight[val_idx]

    print(f"Training samples: {num_train}, Validation samples: {num_val}")

    # --- Compute total parameters dynamically ---
    total_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"Total parameters: {total_params}")

    # --- Total steps across all epochs ---
    total_steps = num_train * epochs

    # --- Memmaps ---
    weight_mm = np.memmap(memmap_path_weights, dtype=np.float32, mode="w+", shape=(total_steps, total_params))
    input_mm = np.memmap(memmap_path_inputs, dtype=np.float32, mode="w+", shape=(total_steps, input_size))
    output_mm = np.memmap(memmap_path_outputs, dtype=np.float32, mode="w+", shape=(total_steps, output_size))

    # --- Belief consistency loss ---
    def belief_consistency_loss(h):
        """
        Encourage hidden activations to be stable within batch
        """
        mean = tf.reduce_mean(h, axis=0, keepdims=True)
        return tf.reduce_mean(tf.square(h - mean))  # MSE from mean

    # --- Training step ---
    @tf.function
    def train_step(x, y, w):
        with tf.GradientTape() as tape:
            y_pred, h = model(x, training=True)
            per_neuron_loss = tf.keras.backend.binary_crossentropy(y, y_pred)
            task_loss = tf.reduce_mean(per_neuron_loss * w)
            belief_loss = belief_consistency_loss(h)
            total_loss = task_loss + lambda_belief * belief_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, task_loss, belief_loss

    # --- Pack weights into 1D tensor ---
    @tf.function
    def pack_weights():
        return tf.concat([tf.reshape(v, [-1]) for v in model.trainable_variables], axis=0)

    # --- Early stopping ---
    best_val_loss = np.inf
    patience_counter = 0
    step_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # --- Shuffle training data per epoch ---
        epoch_indices = np.random.permutation(num_train)
        train_inputs_epoch = train_inputs[epoch_indices]
        train_outputs_epoch = train_outputs[epoch_indices]
        train_mask_epoch = train_mask[epoch_indices]

        # --- Training loop over batches ---
        for step in range(0, num_train, batch_size):
            x_batch = train_inputs_epoch[step:step + batch_size]
            y_batch = train_outputs_epoch[step:step + batch_size]
            w_batch = train_mask_epoch[step:step + batch_size]

            total_loss, task_loss, belief_loss = train_step(x_batch, y_batch, w_batch)

            # --- Log each sample in the batch separately ---
            batch_size_actual = x_batch.shape[0]
            for i in range(batch_size_actual):
                weight_mm[step_counter, :] = pack_weights().numpy()
                input_mm[step_counter, :] = x_batch[i]
                output_mm[step_counter, :] = y_batch[i]
                step_counter += 1

            # --- Print progress ---
            if step_counter % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Step {step_counter} | Total Loss {total_loss.numpy():.6f} | "
                      f"Task Loss {task_loss.numpy():.6f} | Belief Loss {belief_loss.numpy():.6f} | "
                      f"Elapsed: {elapsed:.2f}s")

        # --- Validation at end of epoch ---
        val_pred, val_h = model(val_inputs, training=False)
        per_neuron_val_loss = tf.keras.backend.binary_crossentropy(val_outputs, val_pred)
        val_weighted_loss = per_neuron_val_loss * val_mask
        val_loss_val = float(tf.reduce_mean(val_weighted_loss))
        print(f"End of Epoch {epoch + 1} | Validation Loss: {val_loss_val:.6f}")

        # --- Early stopping ---
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # --- Flush memmaps to disk ---
    weight_mm.flush()
    input_mm.flush()
    output_mm.flush()
    print(f"\nTraining complete. Weights saved to: {memmap_path_weights}")
    print(f"Inputs saved to: {memmap_path_inputs}")
    print(f"Outputs saved to: {memmap_path_outputs}")
