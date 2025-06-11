import os
import csv
import matplotlib.pyplot as plt

def main():
    # TODO: Replace hardcoded value with user input.
    num_models = 2

    folder_path = os.path.dirname(os.path.abspath(__file__))

    # Comment/Uncomment to plot the graphs.
    loss_csv_files = [
        "Wikipedia(num_models=2).csv",
    ]

    # Create plots.
    plt.figure(figsize=(10, 6))
    
    for loss_csv_file in loss_csv_files:
        label = loss_csv_file.split(".")[0]

        loss_fpath = os.path.join(
            folder_path,
            "Loss",
            loss_csv_file)
        with open(loss_fpath, "r") as f:
            csv_reader = csv.reader(f)

            global_steps = []
            test_loss_lists = [[] for _ in range(num_models)]
            percentage_diff_list = []

            for row in csv_reader:
                global_steps.append(int(row[0]))
                # test_loss.append(float(row[1]))
                for model_index, loss_val in enumerate(row[1:]):
                    test_loss_lists[model_index].append(float(loss_val))
                
                diff_val = float(row[1]) - float(row[-1])
                avg_val = (float(row[1]) + float(row[-1])) / 2
                percentage_diff = (diff_val / avg_val) * 100
                percentage_diff_list.append(percentage_diff)

        for index, test_loss in enumerate(test_loss_lists):
            plt.plot(global_steps, test_loss, label=label + "(" + str(index) + ")", marker='o')
        
        plt.plot(global_steps, percentage_diff_list, label="% difference between first and last model", marker='o')

    # Add title and labels.
    plt.title('Comparison of Testing Loss over time.')
    plt.xlabel('Global Step')
    plt.ylabel('Test Loss')

    # Add grid.
    plt.grid(True)

    # Add legend.
    plt.legend()

    # Show plot.
    plt.show()

if __name__ == "__main__":
    main()
