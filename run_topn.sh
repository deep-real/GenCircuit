
GRAPH_ROOT="circuits/EAP-IG-inputs_mean-positional_edge_train_kl_divergence"                   # Root directory containing model subfolders
SCRIPT_PATH="/home/yxpengcs/PycharmProjects/vMIB-circuit/viz_graph.py"    # Replace with path to your Python script
COMPONENT_COUNTS=(101)         # List of top-N components to apply

for dir in $GRAPH_ROOT/camelyon17-set2-mean*/; do
    echo "Checking directory: $dir"
    graph_file="${dir}importances.pt"
    echo "Checking file: $graph_file"
    if [ -f "$graph_file" ]; then
        for num in "${COMPONENT_COUNTS[@]}"; do
            echo "Processing $graph_file with top-$num components..."
            python "$SCRIPT_PATH" \
                --graph_paths "$graph_file" \
                --num_components "$num" \
                --resid_str ""
        done
    else
        echo "File not found: $graph_file"
    fi
done
