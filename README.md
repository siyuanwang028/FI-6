import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price, d1, d2

def calculate_greeks(S, K, T, r, sigma, option_type, d1, d2):
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                  r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    theta = theta_call if option_type == "call" else theta_put
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    rho = rho_call if option_type == "call" else rho_put

    return delta, gamma, vega, theta, rho

def main():
    print("Options Price Calculator")
    S = float(input("Enter current stock price: "))        # 100
    K = float(input("Enter strike price: "))               # 100
    days = int(input("Enter time to expiration (days): ")) # 30
    sigma = float(input("Enter volatility (%): ")) / 100   # 20%
    r = float(input("Enter risk-free rate (%): ")) / 100   # 5%
    option_type = input("Enter option type (call/put): ").strip().lower()  # call

    T = days / 365  # Convert days to years

    price, d1, d2 = black_scholes_price(S, K, T, r, sigma, option_type)
    delta, gamma, vega, theta, rho = calculate_greeks(S, K, T, r, sigma, option_type, d1, d2)

    print("\n== Options Calculator Results ==")
    print(f"Option Price: {price:.4f}")
    print("\nGreeks:")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Vega: {vega:.4f}")
    print(f"Theta: {theta:.4f}")
    print(f"Rho: {rho:.4f}")

if __name__ == "__main__":
    main()
