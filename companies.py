import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Companies:
    def __init__(self, filename, sample_size=None, seed: int = 42) -> None:
        """
        Class constructor. DO NOT MODIFY.

        Args:
            filename: The name of the file containing the data set
            sample_size: The size of the data set to be used, None as default
            seed: The seed for the random number generator

        Returns:
            None
        """

        if sample_size is not None:
            np.random.seed(seed)
            self.df = self.read_data(filename).sample(sample_size)
        else:
            self.df = self.read_data(filename)

    def read_data(self, filename: str) -> pd.DataFrame:
        """
        Reads the data from the given file path.

        Args:
            filename: The path to the file to be read

        Returns:
            A pandas DataFrame containing the data

        """
        # >> YOUR CODE HERE
        csv = pd.read_csv(filename)
        return csv
        # END OF YOUR CODE <<

    def median_revenue(self) -> np.array:
        """
        Calculates the median revenue of companies in each industry, and then save 
        just the industries in a numpy array, sorted by their corresponding median 
        revenue in descending order.

        Returns:
            A numpy array containing the sorted industry names.
        """
        # >> YOUR CODE HERE
        group = self.df.groupby(["industry"])["revenue"].median()
        df_grouped = group.to_frame('median').reset_index()
        df_grouped.sort_values(by=['median'], ascending=False)
        industry_list = df_grouped['industry'].values.tolist()
        final = np.array(industry_list)
        
        
        return final
        # END OF YOUR CODE <<

    def growth_stats_given_workers(self, workers: int) -> tuple:
        """
        Calculates the mean and the standard deviation of the growth of companies 
        that have at least a certain number of workers.

        Args:
            workers: The minimum number of workers for a company to be included in the calculation.

        Returns:
            A python tuple of two floats representing the mean and the sample standard deviation.
        """
        # >> YOUR CODE HERE
        new_df = self.df[self.df['workers' >= workers]]
        
        mean = new_df['growth'].mean()
        std = new_df['growth'].std()
        # END OF YOUR CODE <<

        return float(mean), float(std)

    def plot_histogram(self) -> plt.Figure:
        """
        Generates a histogram of the revenue of companies in California (state code CA).

        The plot uses 15 bins, and the log of the counts to show the comparison more clearly.

        Returns:
            A matplotlib figure of the histogram.
        """
        fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
        # >> YOUR CODE HERE
        new_df = self.df[self.df['state_s'] == 'CA']                            
        ax.hist(new_df['revenue'], bins=15, log=True)
        ax.set_xlabel('Revenue of Companies in California')
        ax.set_ylabel('Number of Companies (Logarithmic)')
        ax.set_title('Distribution of California Companies by Revenue')
        # END OF YOUR CODE <<
        return fig

    def make_rank_plot(self) -> plt.Figure:
        """
        Generates a simple line plot of the rank of each company versus the log of the growth 
        of that company.

        Returns:
            A matplotlib figure of the line plot.
        """
        fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
        # >> YOUR CODE HERE
        ax.plot(self.df['rank'], self.df['growth'])
        ax.set_yscale('log')
        ax.set_xlabel('Rank of Company')
        ax.set_ylabel('Logarithmic Growth of Company')
        ax.set_title('Logarithmic Growth vs. Rank of Company')
        # END OF YOUR CODE <<
        return fig


"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""

from random import seed
import os

    
def evaluate_companies():
    """
    Test your implementation in companies.py.

    Args:
        None

    Returns:
        None
    """

    print('\n\n-------------Companies Dataset-------------\n')
    print('This test is not exhaustive by any means. It only tests if')
    print('your implementation runs without errors.\n')

    companies = Companies(os.path.join(os.path.dirname(__file__), "dataset/companies.csv"))

    fig = companies.plot_histogram()
    fig.savefig(os.path.join(os.path.dirname(__file__), "companies_histogram.png"))

    fig = companies.make_rank_plot()
    fig.savefig(os.path.join(os.path.dirname(__file__), "companies_rank_plot.png"))

    print('Test companies.py: passed')


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    evaluate_companies()