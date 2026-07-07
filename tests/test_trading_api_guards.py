import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import trading_api


class TradingApiLiveGuardTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.copy()
        for key in (
            'EXCHANGE_API_KEY',
            'EXCHANGE_API_SECRET',
            'EXCHANGE_PASSPHRASE',
            'LIVE_TRADING_CONFIRMED',
        ):
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_config_bool_parses_string_false(self):
        self.assertFalse(trading_api._config_bool('false', True))
        self.assertFalse(trading_api._config_bool('0', True))
        self.assertTrue(trading_api._config_bool('true', False))
        self.assertTrue(trading_api._config_bool('1', False))

    def test_live_requires_explicit_confirmation_before_dependency_imports(self):
        with self.assertRaisesRegex(ValueError, 'explicit confirmation'):
            trading_api._build_bot(
                {'exchange': 'binance', 'symbols': ['BTC/USDT'], 'paper': 'false'},
                lambda _: None,
            )

    def test_live_requires_key_and_secret_after_confirmation(self):
        with self.assertRaisesRegex(ValueError, 'requires EXCHANGE_API_KEY'):
            trading_api._build_bot(
                {
                    'exchange': 'binance',
                    'symbols': ['BTC/USDT'],
                    'paper': False,
                    'live_confirmed': True,
                },
                lambda _: None,
            )

    def test_okx_live_requires_passphrase(self):
        os.environ['EXCHANGE_API_KEY'] = 'key'
        os.environ['EXCHANGE_API_SECRET'] = 'secret'
        with self.assertRaisesRegex(ValueError, 'EXCHANGE_PASSPHRASE'):
            trading_api._build_bot(
                {
                    'exchange': 'okx',
                    'symbols': ['BTC/USDT'],
                    'paper': False,
                    'live_confirmed': True,
                },
                lambda _: None,
            )


if __name__ == '__main__':
    unittest.main()
