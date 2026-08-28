# analyzer.py
import os
import time
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class PoliticalSentimentAnalyzer:

  def __init__(self):
    self.vader = SentimentIntensityAnalyzer()
    # Unique User-Agent avoids triggering shared IP rate-limiting
    self.headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'PoliticalSentimentBot/1.0 (Student Project)'
        ),
        'Accept': (
            'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        ),
        'Accept-Language': 'en-US,en;q=0.5',
    }

  def fetch_subreddit_posts_keyless(
      self, subreddit_name: str, limit: int = 50
  ) -> pd.DataFrame:
    """Fetch recent posts from a subreddit using Reddit's RSS feed (bypasses 403 JSON blocks)."""
    url = f'https://www.reddit.com/r/{subreddit_name}/new.rss?limit={limit}'

    try:
      response = requests.get(url, headers=self.headers, timeout=10)
      if response.status_code != 200:
        print(
            f'⚠️ Failed to fetch r/{subreddit_name}: HTTP Status'
            f' {response.status_code}'
        )
        return pd.DataFrame()

      # Parse Atom XML content from RSS feed
      root = ET.fromstring(response.content)
      ns = {'atom': 'http://www.w3.org/2005/Atom'}

      posts_data = []
      for entry in root.findall('atom:entry', ns):
        title_elem = entry.find('atom:title', ns)
        content_elem = entry.find('atom:content', ns)
        id_elem = entry.find('atom:id', ns)

        title = title_elem.text if title_elem is not None else ''
        content = content_elem.text if content_elem is not None else ''
        post_id = id_elem.text if id_elem is not None else ''

        full_text = f'{title} {content}'.strip()

        posts_data.append({
            'id': post_id,
            'subreddit': subreddit_name,
            'created_utc': pd.Timestamp.now(),
            'title': title,
            'text': full_text,
            'score': 1,
            'num_comments': 0,
        })

      return pd.DataFrame(posts_data)

    except Exception as e:
      print(f'Error fetching r/{subreddit_name}: {e}')
      return pd.DataFrame()

  def analyze_entity_sentiment(
      self, df: pd.DataFrame, entities: dict
  ) -> pd.DataFrame:
    """Score sentiment for posts mentioning specific entity keywords."""
    results = []

    for _, row in df.iterrows():
      text_lower = row['text'].lower()

      for entity_label, keywords in entities.items():
        if any(kw in text_lower for kw in keywords):
          sentiment_scores = self.vader.polarity_scores(row['text'])

          results.append({
              'post_id': row['id'],
              'subreddit': row['subreddit'],
              'created_at': row['created_utc'],
              'title': row['title'],
              'entity': entity_label,
              'compound_score': sentiment_scores['compound'],
              'score_weight': row['score'],
              'num_comments': row['num_comments'],
          })

    return pd.DataFrame(results)


if __name__ == '__main__':
  analyzer = PoliticalSentimentAnalyzer()

  print('Fetching recent posts from Reddit RSS feeds...')
  df_cons = analyzer.fetch_subreddit_posts_keyless('Conservative', limit=50)

  print('Pausing to respect rate limits...')
  time.sleep(4)  # 4-second delay prevents HTTP 429 errors

  df_dems = analyzer.fetch_subreddit_posts_keyless('democrats', limit=50)

  if df_cons.empty and df_dems.empty:
    print('❌ No data retrieved. Check your network connection.')
  else:
    df_combined = pd.concat([df_cons, df_dems], ignore_index=True)

    target_entities = {
        'Trump / MAGA': ['trump', 'maga'],
        'Democrats': ['democrat', 'democrats', 'harris', 'biden'],
    }

    print('Analyzing sentiment across targeted entities...')
    sentiment_df = analyzer.analyze_entity_sentiment(
        df_combined, target_entities
    )

    os.makedirs('data', exist_ok=True)
    output_path = 'data/sentiment_results.csv'
    sentiment_df.to_csv(output_path, index=False)

    print(f'✅ Success! Processed {len(sentiment_df)} entity mentions.')
    print(f'📁 Results saved to: {output_path}')