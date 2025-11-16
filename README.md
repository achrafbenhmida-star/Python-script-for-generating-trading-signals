# Python-script-for-generating-trading-signals
This Python script is a complete tool for Forex market analysis. It downloads historical data, calculates technical indicators, generates trading signals, detects chart patterns, manages risk, and visualizes results. The documentation explains its features and code structure.
# Importation des bibliothéques
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import matplotlib.pyplot as plt
import logging
# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 1. Téléchargement des données
class DataDownloader:
    @staticmethod
    def download_data(symbol, period='3mo', interval='1h'):
        try:
            df = yf.download(symbol, period=period, interval=interval)
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement des données pour {symbol}: {e}")
            return pd.DataFrame()
        
        if df.empty:
            return pd.DataFrame()

        df.reset_index(inplace=True)
        df.rename(columns={'Datetime': 'Timestamp'}, inplace=True)
        return df
        # 2. Calcul des indicateurs techniques
class TechnicalIndicators:
    @staticmethod
    def calculate_indicators(data):
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['Stoch_K'] = stoch.stoch()
        data['Stoch_D'] = stoch.stoch_signal()
        macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['Short_EMA'] = ta.trend.EMAIndicator(data['Close'], window=9).ema_indicator()
        data['Long_EMA'] = ta.trend.EMAIndicator(data['Close'], window=21).ema_indicator()
        return data
        # 3. Génération des signaux de trading
class TradingSignals:
    @staticmethod
    def generate_signals(data):
        if len(data) < 2:
            return "NEUTRAL"
        
        buy_signal = (
            (data['Short_EMA'].iloc[-1] > data['Long_EMA'].iloc[-1]) &
            (
                ((data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]) & (data['MACD'].iloc[-2] < data['MACD_Signal'].iloc[-2])) |
                (data['RSI'].iloc[-1] > 70) |
                ((data['RSI'].iloc[-1] > 30) & (data['RSI'].iloc[-2] < 30)) |
                (data['Stoch_K'].iloc[-1] > 80) |
                ((data['Stoch_K'].iloc[-1] > 20) & (data['Stoch_K'].iloc[-2] < 20)) |
                ((data['Close'].iloc[-1] > data['BB_Lower'].iloc[-1]) & (data['Stoch_K'].iloc[-1] > 20) & (data['Stoch_K'].iloc[-2] < 20))
            )
        )
        sell_signal = (
            (data['Short_EMA'].iloc[-1] < data['Long_EMA'].iloc[-1]) &
            (
                ((data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1]) & (data['MACD'].iloc[-2] > data['MACD_Signal'].iloc[-2])) |
                (data['RSI'].iloc[-1] < 30) |
                ((data['RSI'].iloc[-1] < 70) & (data['RSI'].iloc[-2] > 70)) |
                (data['Stoch_K'].iloc[-1] < 20) |
                ((data['Stoch_K'].iloc[-1] < 80) & (data['Stoch_K'].iloc[-2] > 80)) |
                ((data['Close'].iloc[-1] < data['BB_Upper'].iloc[-1]) & (data['Stoch_K'].iloc[-1] < 80) & (data['Stoch_K'].iloc[-2] > 80))
            )
        )
        
        if buy_signal:
            return "ACHETER"
        elif sell_signal:
            return "VENDRE"
        else:
            return "NEUTRAL"
            # 4. Détection des modèles de graphiques
class PatternDetection:
    @staticmethod
    def detect_patterns(data):
        patterns = []
        open = data['Open']
        close = data['Close']
        high = data['High']
        low = data['Low']

        # Détection des modèles de retournement
        if len(close) >= 5:
            if close.iloc[-3] > close.iloc[-4] and close.iloc[-3] > close.iloc[-2] and close.iloc[-2] < close.iloc[-1]:
                patterns.append("Double Top")
            if close.iloc[-3] < close.iloc[-4] and close.iloc[-3] < close.iloc[-2] and close.iloc[-2] > close.iloc[-1]:
                patterns.append("Double Bottom")

        if len(close) >= 7:
            if close.iloc[-5] > close.iloc[-6] and close.iloc[-5] > close.iloc[-4] and close.iloc[-3] > close.iloc[-4] and close.iloc[-3] > close.iloc[-2] and close.iloc[-1] < close.iloc[-2]:
                patterns.append("Triple Top")
            if close.iloc[-5] < close.iloc[-6] and close.iloc[-5] < close.iloc[-4] and close.iloc[-3] < close.iloc[-4] and close.iloc[-3] < close.iloc[-2] and close.iloc[-1] > close.iloc[-2]:
                patterns.append("Triple Bottom")
            if close.iloc[-6] < close.iloc[-5] and close.iloc[-4] < close.iloc[-3] and close.iloc[-2] < close.iloc[-1] and close.iloc[-5] > close.iloc[-3] and close.iloc[-3] > close.iloc[-1]:
                patterns.append("Head and Shoulders")
            if close.iloc[-6] > close.iloc[-5] and close.iloc[-4] > close.iloc[-3] and close.iloc[-2] > close.iloc[-1] and close.iloc[-5] < close.iloc[-3] and close.iloc[-3] < close.iloc[-1]:
                patterns.append("Inverse Head and Shoulders")

        # Détection des modèles de continuation
        # 4.2.1 Rectangles
        if len(close) >= 10:
            if max(high[-10:]) == max(high[-5:]) and min(low[-10:]) == min(low[-5:]):
                if close.iloc[-1] < min(low[-5:]):
                    patterns.append("Rectangle Baissier")
                elif close.iloc[-1] > max(high[-5:]):
                    patterns.append("Rectangle Haussier")
        
        # 4.2.2 Drapeaux
        if len(close) >= 7:
            if close.iloc[-1] < close.iloc[-5] and min(low[-5:-1]) > min(low[-7:-5]):
                patterns.append("Drapeau Baissier")
            if close.iloc[-1] > close.iloc[-5] and max(high[-5:-1]) < max(high[-7:-5]):
                patterns.append("Drapeau Haussier")
        
        # 4.2.3 Pennants
        if len(close) >= 7:
            if close.iloc[-1] < close.iloc[-5] and max(high[-7:-1]) < max(high[-10:]) and min(low[-7:-1]) > min(low[-10:]):
                patterns.append("Pennant Baissier")
            if close.iloc[-1] > close.iloc[-5] and max(high[-7:-1]) > max(high[-10:]) and min(low[-7:-1]) < min(low[-10:]):
                patterns.append("Pennant Haussier")

        # 4.2.4 Cup & Handle
        if len(close) >= 15:
            cup_high = max(close[-15:])
            cup_low = min(close[-15:])
            handle_high = max(close[-7:])
            handle_low = min(close[-7:])
            if (cup_high == max(close[-15:]) and 
                cup_low == min(close[-15:]) and
                handle_high < cup_high and 
                handle_low > cup_low):
                if close.iloc[-1] > handle_high:
                    patterns.append("Cup & Handle")
                if close.iloc[-1] < handle_low:
                    patterns.append("Inverse Cup & Handle")
        
        # Détection des modèles de chandeliers
        if len(close) >= 3:
            # Marteau et Marteau Inversé
            if close.iloc[-1] > open.iloc[-1] and (high.iloc[-1] - low.iloc[-1]) > 2 * abs(open.iloc[-1] - close.iloc[-1]) and (close.iloc[-1] - low.iloc[-1]) / (.001 + high.iloc[-1] - low.iloc[-1]) > 0.6:
                patterns.append("Marteau")
            if open.iloc[-1] > close.iloc[-1] and (high.iloc[-1] - low.iloc[-1]) > 2 * abs(open.iloc[-1] - close.iloc[-1]) and (high.iloc[-1] - open.iloc[-1]) / (.001 + high.iloc[-1] - low.iloc[-1]) > 0.6:
                patterns.append("Marteau Inversé")

            # Étoile du Matin
            if open.iloc[-3] > close.iloc[-3] and open.iloc[-2] > close.iloc[-2] and close.iloc[-1] > open.iloc[-1] and close.iloc[-1] > (open.iloc[-3] + close.iloc[-3]) / 2:
                patterns.append("Étoile du Matin")

            # Étoile du Soir
            if close.iloc[-3] > open.iloc[-3] and close.iloc[-2] < open.iloc[-2] and open.iloc[-1] > close.iloc[-1] and close.iloc[-1] < (open.iloc[-3] + close.iloc[-3]) / 2:
                patterns.append("Étoile du Soir")

            # Trois Soldats Blancs
            if close.iloc[-3] < open.iloc[-3] and close.iloc[-2] < open.iloc[-2] and close.iloc[-1] > open.iloc[-1] and close.iloc[-2] > close.iloc[-3] and close.iloc[-1] > close.iloc[-2]:
                patterns.append("Trois Soldats Blancs")

            # Trois Corbeaux Noirs
            if close.iloc[-3] > open.iloc[-3] and close.iloc[-2] > open.iloc[-2] and close.iloc[-1] < open.iloc[-1] and close.iloc[-2] < close.iloc[-3] and close.iloc[-1] < close.iloc[-2]:
                patterns.append("Trois Corbeaux Noirs")

            # Harami
            if close.iloc[-2] > open.iloc[-2] and open.iloc[-1] < close.iloc[-1] and close.iloc[-1] < close.iloc[-2] and open.iloc[-1] > open.iloc[-2]:
                patterns.append("Harami Baissier")
            if close.iloc[-2] < open.iloc[-2] and open.iloc[-1] > close.iloc[-1] and close.iloc[-1] > close.iloc[-2] and open.iloc[-1] < open.iloc[-2]:
                patterns.append("Harami Haussier")

            # Marubozu
            if open.iloc[-1] == low.iloc[-1] and close.iloc[-1] == high.iloc[-1]:
                patterns.append("Marubozu Haussier")
            if open.iloc[-1] == high.iloc[-1] and close.iloc[-1] == low.iloc[-1]:
                patterns.append("Marubozu Baissier")

        if patterns:
            return [patterns[0]]  # Return only the first detected pattern
        return patterns
        # 5. Gestion des risques
class RiskManagement:
    @staticmethod
    def calculate_risk_management(data, entry_price, signal, risk_percentage, balance):
        atr = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range().iloc[-1]
        if signal == "ACHETER":
            stop_loss = entry_price - atr
            take_profit = entry_price + 3 * atr
        elif signal == "VENDRE":
            stop_loss = entry_price + atr
            take_profit = entry_price - 3 * atr
        else:
            return None, None

        return round(stop_loss, 5), round(take_profit, 5)
        # 6. Visualisation des données
class ChartPlotter:
    @staticmethod
    def plot_chart(data, symbol, entry_price, stop_loss=None, take_profit=None):
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(14, 8))
        
        # Plotting closing prices
        plt.plot(data['Close'], label='Prix de clôture', color='blue', linewidth=2)
        
        # Plotting EMAs
        plt.plot(data['Short_EMA'], label='EMA Court terme', color='red', linewidth=2)
        plt.plot(data['Long_EMA'], label='EMA Long terme', color='green', linewidth=2)
        
        # Plotting Bollinger Bands
        plt.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], color='gray', alpha=0.3, label='Bandes de Bollinger')
        
        # Annotating entry, stop-loss, and take-profit prices without amounts
        if stop_loss is not None and take_profit is not None:
            plt.axhline(y=stop_loss, color='r', linestyle='--', linewidth=1.5, label='Stop-Loss')
            plt.axhline(y=take_profit, color='g', linestyle='--', linewidth=1.5, label='Take-Profit')
            plt.axhline(y=entry_price, color='b', linestyle='--', linewidth=1.5, label='Prix d\'entrée')
        
        plt.legend()
        plt.title(f'Graphique pour {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Prix')
        plt.grid(True)
        plt.show()
        # 7. Analyse du Forex
class ForexAnalyzer:
    def __init__(self, symbols, time_frames, balance=10000, risk_percentage=0.01):
        self.symbols = symbols
        self.time_frames = time_frames
        self.balance = balance
        self.risk_percentage = risk_percentage

    def analyze(self):
        for symbol in self.symbols:
            for time_frame in self.time_frames:
                logging.info(f"Analyzing {symbol} ({time_frame} time frame)...")
                data = DataDownloader.download_data(symbol, period='1mo', interval=time_frame)
                
                if data.empty:
                    logging.warning(f"Pas de données disponibles pour {symbol}.")
                    continue

                data = TechnicalIndicators.calculate_indicators(data)
                signal = TradingSignals.generate_signals(data)
                if signal in ["ACHETER", "VENDRE"]:
                    logging.info(f"Signal pour {symbol} ({time_frame}): {signal}")

                    patterns = PatternDetection.detect_patterns(data)
                    if patterns:
                        logging.info(f"Pattern détecté pour {symbol} ({time_frame}): {patterns[0]}")

                    entry_price = data['Close'].iloc[-1]
                    stop_loss, take_profit = RiskManagement.calculate_risk_management(data, entry_price, signal, self.risk_percentage, self.balance)

                    if stop_loss is not None and take_profit is not None:
                        if 'JPY' in symbol:
                            entry_price = round(entry_price, 3)
                            stop_loss = round(stop_loss, 3)
                            take_profit = round(take_profit, 3)
                        else:
                            entry_price = round(entry_price, 5)
                            stop_loss = round(stop_loss, 5)
                            take_profit = round(take_profit, 5)
                        
                        logging.info(f"Entrée: {entry_price}, Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")

                        ChartPlotter.plot_chart(data, symbol, entry_price, stop_loss, take_profit)
                    else:
                        logging.warning(f"Calcul de gestion des risques impossible pour {symbol} ({time_frame})")
                else:
                    logging.info(f"Aucun signal de trading pour {symbol} ({time_frame})")

if __name__ == "__main__":
    MAJEURES = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X']
    MINEURES = ['EURGBP=X', 'EURCHF=X', 'EURAUD=X', 'EURCAD=X', 'EURNZD=X', 'GBPCHF=X', 'GBPAUD=X', 'GBPCAD=X', 'GBPNZD=X', 'AUDCAD=X', 'AUDNZD=X', 'CADCHF=X', 'NZDCHF=X', 'AUDCHF=X']
    EXOTIQUES = ['USDTND=X', 'USDSAR=X', 'USDQAR=X', 'USDAED=X', 'USDTRY=X']
    ALL_PAIRS = MAJEURES + MINEURES + EXOTIQUES

    time_frames = ['5m', '15m', '30m', '1h']
    analyzer = ForexAnalyzer(ALL_PAIRS, time_frames)
    analyzer.analyze()
