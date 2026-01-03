import time

import numpy as np
import tensorflow as tf


def train_with_full_weight_logging(
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
        print_every=10000,
        early_stop_patience=10,
        random_seed=None
):
    """
    Custom training loop with:
      - per-neuron mask (sample_weight per output neuron)
      - per-step weight logging to memmap
      - per-step input/output logging to memmap
      - random shuffling per epoch
      - early stopping based on validation loss
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

    # --- Training step ---
    @tf.function
    def train_step(x, y, w):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            per_neuron_loss = tf.keras.backend.binary_crossentropy(y, y_pred)
            weighted_loss = tf.reduce_mean(per_neuron_loss * w)
        grads = tape.gradient(weighted_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return weighted_loss

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

        # --- Training loop ---
        for step in range(num_train):
            x = train_inputs_epoch[step:step + 1]
            y = train_outputs_epoch[step:step + 1]
            w = train_mask_epoch[step:step + 1]

            loss = train_step(x, y, w)

            # --- Log weights and corresponding input/output ---
            weight_mm[step_counter, :] = pack_weights().numpy()
            input_mm[step_counter, :] = x
            output_mm[step_counter, :] = y

            step_counter += 1

            # --- Print progress ---
            if step_counter % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Step {step_counter} | Loss {loss.numpy():.6f} | Elapsed: {elapsed:.2f}s")

        # --- Validation at end of epoch ---
        val_pred = model(val_inputs, training=False)
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
