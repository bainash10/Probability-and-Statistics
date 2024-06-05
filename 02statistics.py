import statistics as stats

def overview_basic_statistics():
    data = [2, 4, 4, 4, 5, 5, 7, 9, 10]
    
    print("Data:", data)
    
    # Basic statistics with the statistics module
    mean = stats.mean(data)
    median = stats.median(data)
    mode = stats.mode(data)
    variance = stats.variance(data)
    stdev = stats.stdev(data)
    
    print("\nBasic Statistics:")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Variance: {variance}")
    print(f"Standard Deviation: {stdev}")

# Call the function to display the overview
overview_basic_statistics()
