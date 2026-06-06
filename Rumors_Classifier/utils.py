import os

def parse_propagation_file(filepath):
    """
    Parses a propagation file (replies or retweets) that uses ##id## blocks.
    Returns a dict: {tweet_id: count_of_replies_or_retweets}
    """
    counts = {}
    current_id = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('##') and line.endswith('##'):
                # New parent ID block
                current_id = line[2:-2]   # remove the double hashes
                counts[current_id] = 0
            else:
                # This is a reply/retweet ID belonging to current_id
                if current_id is not None:
                    counts[current_id] += 1
    return counts


