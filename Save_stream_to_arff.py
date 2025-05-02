from copy import deepcopy
from river.datasets import synth

def save_stream_to_arff(stream, relation_name="agrawal_stream", output_file="agrawal_stream.arff"):
    stream = list(stream)  # Unpack the stream to inspect
    if not stream:
        raise ValueError("Stream is empty")

    x_example, _ = stream[0]
    feature_names = list(x_example.keys())
    class_labels = sorted(set(y for _, y in stream))

    with open(output_file, "w") as f:
        # Header
        f.write(f"@RELATION {relation_name}\n\n")

        for feat in feature_names:
            f.write(f"@ATTRIBUTE {feat} NUMERIC\n")  # You can customize this if needed

        f.write(f"@ATTRIBUTE class {{{', '.join(map(str, class_labels))}}}\n\n")
        f.write("@DATA\n")

        # Data rows
        for x, y in stream:
            values = [str(x[feat]) for feat in feature_names]
            values.append(str(y))
            f.write(",".join(values) + "\n")

    print(f"Stream saved to {output_file} in ARFF format.")

# Usage
# stream = deepcopy(synth.Agrawal()).take(1000)
# save_stream_to_arff(stream)
