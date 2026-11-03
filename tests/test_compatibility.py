from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean

ROOT = Path(__file__).resolve().parents[1]
for directory in ('02_making_recommendations', '03_discovering_groups', '04_searching_and_ranking'):
    sys.path.insert(0, str(ROOT / directory))

import clusters
import recommendations
from searchengine import Crawler


@contextmanager
def local_page():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = '<html><body>recommendation clustering</body></html>'.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:%s/' % server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class CompatibilityTests(unittest.TestCase):
    def test_recommendations_with_scipy(self):
        ranks = recommendations.recommendations_by_people(recommendations.critics, 'Toby')
        self.assertEqual(ranks[0][1], 'The Night Listener')
        self.assertAlmostEqual(ranks[0][0], 3.3477895267)
        items = recommendations.calculate_similar_items(
            recommendations.critics, similarity=recommendations.sim_euclidean)
        item_ranks = recommendations.recommendations_by_items(recommendations.critics, items, 'Toby')
        self.assertTrue(item_ranks)
        self.assertTrue(all(np.isfinite(score) for score, _ in item_ranks))

    def test_kmeans_accepts_unequal_cluster_sizes(self):
        with patch('clusters.random.random', side_effect=[0.0, 1.0]):
            result = clusters.k_mean_classify(np.array([[0., 0., 0., 10.]]), euclidean, k=2)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.tolist(), [[0, 1, 2], [3]])

    def test_kmeans_preserves_empty_cluster_centres(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with patch('clusters.random.random', side_effect=[0.0, 0.0, 1.0]):
                result = clusters.k_mean_classify(
                    np.array([[0., 0., 0., 10.]]), euclidean, k=3, convert=str)
        self.assertEqual(result, [['0', '1', '2'], [], ['3']])

    def test_kmeans_accepts_equal_cluster_sizes(self):
        with patch('clusters.random.random', side_effect=[0.0, 1.0]):
            result = clusters.k_mean_classify(np.array([[0., 1., 9., 10.]]), euclidean, k=2)
        self.assertEqual(result.tolist(), [[0, 1], [2, 3]])

    def test_dendrogram_is_a_readable_jpeg(self):
        tree = clusters.hierarchical_classify(np.array([[0., 1., 10.], [0., 2., 9.]]), euclidean)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'clusters.jpg')
            with patch.object(clusters, 'titles', ['a', 'b', 'c'], create=True):
                clusters.plot_tree(tree, width=400, convert=lambda i: ['a', 'b', 'c'][i], filename=filename)
            with Image.open(filename) as image:
                self.assertEqual(image.format, 'JPEG')
                self.assertEqual(image.size, (400, 60))
                image.load()
                self.assertIsNotNone(image.getbbox())

    def test_local_http_page_is_parsed_tokenized_and_indexed(self):
        crawler = Crawler(':memory:')
        self.addCleanup(crawler.conn.close)
        crawler.create_tables()
        with local_page() as url:
            crawler.crawl([url], depth=1)
        self.assertEqual(crawler.conn.execute('SELECT url FROM urls').fetchone()[0], url)
        words = {row[0] for row in crawler.conn.execute('SELECT word FROM words')}
        self.assertTrue({'recommendation', 'clustering'}.issubset(words))
        self.assertGreater(crawler.conn.execute('SELECT count(*) FROM url_word_refs').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
