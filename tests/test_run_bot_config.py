import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import run_bot


class RunBotConfigTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.copy()
        for key in (
            'PAPER_TRADING',
            'LIVE_TRADING_CONFIRMED',
            'EXCHANGE',
            'EXCHANGE_API_KEY',
            'EXCHANGE_API_SECRET',
            'EXCHANGE_PASSPHRASE',
        ):
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_env_bool_parses_false_values(self):
        os.environ['PAPER_TRADING'] = 'false'
        self.assertFalse(run_bot.build_config()['paper'])
        os.environ['PAPER_TRADING'] = '0'
        self.assertFalse(run_bot.build_config()['paper'])
        os.environ['PAPER_TRADING'] = 'off'
        self.assertFalse(run_bot.build_config()['paper'])

    def test_live_requires_railway_safe_confirmation(self):
        config = run_bot.build_config() | {'paper': False, 'exchange': 'binance'}
        with self.assertRaisesRegex(ValueError, 'LIVE_TRADING_CONFIRMED=true'):
            run_bot.validate_config(config)

    def test_live_requires_exchange_credentials(self):
        os.environ['LIVE_TRADING_CONFIRMED'] = 'true'
        config = run_bot.build_config() | {'paper': False, 'exchange': 'binance'}
        with self.assertRaisesRegex(ValueError, 'EXCHANGE_API_KEY'):
            run_bot.validate_config(config)

    def test_okx_live_requires_passphrase(self):
        os.environ['LIVE_TRADING_CONFIRMED'] = 'true'
        config = run_bot.build_config() | {
            'paper': False,
            'exchange': 'okx',
            'api_key': 'key',
            'api_secret': 'secret',
            'api_passphrase': '',
        }
        with self.assertRaisesRegex(ValueError, 'EXCHANGE_PASSPHRASE'):
            run_bot.validate_config(config)


if __name__ == '__main__':
    unittest.main()
