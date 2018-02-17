import sqlite3
import datetime
import multiprocessing
from urllib.parse import urljoin

import requests

from bs4 import BeautifulSoup

import jieba


class Crawler:
    """简单爬虫"""

    def __init__(self, database) -> None:
        self.conn = sqlite3.connect(database)

    def __del__(self) -> None:
        self.conn.close()

    def get_entry_id(self, table, field, value, create_if_not_exist=True, **kwargs):
        param = (table, field, "'%s'" % value if isinstance(value, str) else value)
        result = self.conn.execute('SELECT rowid FROM %s WHERE %s=%s;' % param).fetchone()
        row_id = None if result is None else result[0]

        if row_id is None and create_if_not_exist:
            fields, values = [field], [value]
            for field, value in kwargs.items():
                fields.append(field)
                values.append(value)
            param = (table, ', '.join(fields), ', '
                     .join(["'%s'" % value if isinstance(value, str) else value for value in values]))
            row_id = self.conn.execute('INSERT INTO %s (%s) VALUES (%s);' % param).lastrowid

        return row_id

    def add_to_index(self, url, soup):
        if not self.should_be_index(url):
            return

        print('Indexing %s...' % url)

        content = soup.get_text(strip=True)
        words = jieba.tokenize(content, mode='search')
        url_id = self.get_entry_id('urls', 'url', url, content=content)

        for word in words:
            word_id = self.get_entry_id('words', 'word', word[0])
            self.conn.execute('INSERT INTO url_word_refs (url_id, word_id, location) VALUES (%s, %s, %s);'
                              % (url_id, word_id, word[1]))

    def should_be_index(self, url, expired_days=7):
        result = self.conn.execute("SELECT rowid, last_update FROM urls WHERE url='%s';" % url).fetchone()
        if result is not None:
            url_id, last_update = result[0], datetime.datetime.fromtimestamp(result[1])
            if last_update > datetime.datetime.now() - datetime.timedelta(days=expired_days):
                word_id = self.get_entry_id('url_word_refs', 'url_id', url_id, create_if_not_exist=False)
                if word_id is not None:
                    return False
        return True

    def add_link_ref(self, url_from, url_to, link_text):
        pass

    def crawl(self, start_urls, depth=2) -> None:
        urls, new_urls = set(start_urls), None
        for _ in range(depth):
            new_urls = set()
            for url in urls:
                try:
                    response = requests.get(url)
                except Exception as e:
                    print('[WARNING] get %s failed, detail:%s.' % (url, e))
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                self.add_to_index(url, soup)

                links = soup.find_all('a')
                for link in links:
                    if 'href' in dict(link.attrs):
                        new_url = urljoin(url, link['href'])
                        if new_url.find("'") >= 0:
                            continue
                        new_url = new_url.split('#')[0]
                        if new_url.startswith('http') and self.should_be_index(new_url):
                            new_urls.add(new_url)
                        self.add_link_ref(url, new_url, link.get_text())
                self.conn.commit()
            urls = new_urls

    def create_tables(self):
        self.conn.execute('CREATE TABLE IF NOT EXISTS urls '
                          '(url, content TEXT, last_update DATETIME DEFAULT CURRENT_TIMESTAMP);')
        self.conn.execute('CREATE TABLE IF NOT EXISTS words (word);')
        self.conn.execute('CREATE TABLE IF NOT EXISTS url_word_refs '
                          '(url_id INTEGER, word_id INTEGER, location INTEGER);')
        self.conn.execute('CREATE TABLE IF NOT EXISTS links (from_id INTEGER, to_id INTEGER);')
        self.conn.execute('CREATE TABLE IF NOT EXISTS link_word_refs (link_id INTEGER, word_id INTEGER);')

        self.conn.execute('CREATE INDEX words__idx ON words (word);')
        self.conn.execute('CREATE INDEX urls__idx ON urls (url);')
        self.conn.execute('CREATE INDEX url_word_refs__idx ON url_word_refs (word_id);')
        self.conn.execute('CREATE INDEX links__from_id__idx ON links (from_id);')
        self.conn.execute('CREATE INDEX links__to_id__idx ON links (to_id);')
        self.conn.execute('CREATE INDEX link_word_refs__idx ON link_word_refs (link_id);')

        self.conn.commit()


class Searcher:

    def __init__(self, database) -> None:
        self.conn = sqlite3.connect(database)

    def __del__(self) -> None:
        self.conn.close()

    def query(self, query):
        words = jieba.cut_for_search(query)



if __name__ == '__main__':
    jieba.enable_parallel(multiprocessing.cpu_count() - 1)
    crawler = Crawler('')
    crawler.crawl(['https://en.wikipedia.org/wiki/Main_Page'])
