import streamlit as st
import pandas as pd
import numpy as np
from vnstock import Vnstock
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import random

# Hàm tạo màu ngẫu nhiên
def random_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)

# Cấu hình giao diện trang web
st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
st.title("Portfolio Optimization Dashboard")
st.write("Ứng dụng tích hợp quy trình: tải dữ liệu cổ phiếu, xử lý, tối ưu hóa danh mục đầu tư (SLSQP, SGD, SGD - Sharpe), so sánh với VN-Index và trực quan hóa dữ liệu.")

pages = [
    "Fetch Stock Data",
    "Portfolio Optimization (SLSQP)",
    "Portfolio Optimization (SGD)",
    "Portfolio Optimization (SGD - Sharpe)",
    "Comparison with VN-Index (SLSQP)",
    "Comparison with VN-Index (SGD)",
    "Comparison with VN-Index (SGD - Sharpe)",
    "Data Visualization",
    "Company Information",
    "Financial Statements"  # New page added
]
page = st.sidebar.radio("Chọn trang", pages)
###########################################
# Trang 1: Fetch Stock Data & Process
###########################################
if page == "Fetch Stock Data":
    st.header("Nhập mã cổ phiếu và tải dữ liệu")
    st.write("Nhập các mã cổ phiếu (phân cách bởi dấu phẩy, ví dụ: ACB, VCB):")
    symbols_input = st.text_input("Mã cổ phiếu", "ACB, VCB")
    
    if st.button("Tải dữ liệu"):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.error("Danh sách mã cổ phiếu không được để trống!")
        else:
            # Save symbols to session state
            st.session_state['symbols'] = symbols
            all_data = []
            for symbol in symbols:
                try:
                    stock = Vnstock().stock(symbol=symbol, source='VCI')
                    historical_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
                    if historical_data.empty:
                        st.warning(f"Không tìm thấy dữ liệu cho mã: {symbol}")
                        continue
                    historical_data['symbol'] = symbol
                    all_data.append(historical_data)
                    st.success(f"Đã tải dữ liệu cho: {symbol}")
                except Exception as e:
                    st.error(f"Lỗi khi tải dữ liệu cho {symbol}: {e}")
            if all_data:
                final_data = pd.concat(all_data, ignore_index=True)
                st.write("Đã kết hợp toàn bộ dữ liệu thành công!")
                
                def calculate_features(data):
                    data['daily_return'] = data['close'].pct_change()
                    data['volatility'] = data['daily_return'].rolling(window=30).std()
                    data.dropna(inplace=True)
                    return data
                
                processed_data = final_data.groupby('symbol').apply(calculate_features)
                processed_data = processed_data.reset_index(drop=True)
                processed_data.to_csv("processed_stock_data.csv", index=False)
                st.success("Dữ liệu xử lý đã được lưu vào file 'processed_stock_data.csv'.")
                st.dataframe(processed_data)
            else:
                st.error("Không có dữ liệu hợp lệ để xử lý!")

###########################################
# Trang 2: Portfolio Optimization (SLSQP)
###########################################
elif page == "Portfolio Optimization (SLSQP)":
    st.header("Tối ưu hóa danh mục đầu tư (SLSQP)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý từ file 'processed_stock_data.csv'.")
    except Exception:
        st.error("Không tìm thấy file 'processed_stock_data.csv'. Vui lòng chuyển đến trang 'Fetch Stock Data' để tải dữ liệu.")
        st.stop()
    
    # Tính toán kỳ vọng lợi nhuận và ma trận hiệp phương sai
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    pivot_returns = processed_data.pivot(index='time', columns='symbol', values='daily_return')
    cov_matrix = pivot_returns.cov()
    
    # Hàm mục tiêu: tối thiểu hóa độ rủi ro (volatility)
    def objective(weights, expected_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(len(expected_returns)))
    total_expected_return = expected_returns.sum()
    initial_weights = expected_returns / total_expected_return
    
    result = minimize(objective, initial_weights, args=(expected_returns, cov_matrix), 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    
    portfolio_df = pd.DataFrame({
        'Stock': expected_returns.index,
        'Optimal Weight': optimal_weights
    })
    st.subheader("Trọng số tối ưu của danh mục (SLSQP):")
    st.dataframe(portfolio_df)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Biểu đồ Donut', 'Biểu đồ Cột'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )
    fig.add_trace(
        go.Pie(
            labels=portfolio_df['Stock'],
            values=portfolio_df['Optimal Weight'],
            hole=0.3,
            textinfo='percent+label',
            marker=dict(
                colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15C', '#19D895', '#F2A900'],
                line=dict(color='#000000', width=2)
            )
        ),
        row=1, col=1
    )
    column_colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2', '#FF1493', '#00FA9A']
    fig.add_trace(
        go.Bar(
            x=portfolio_df['Stock'],
            y=portfolio_df['Optimal Weight'],
            marker=dict(color=column_colors, line=dict(color='#000000', width=2))
        ),
        row=1, col=2
    )
    fig.update_layout(title="So sánh trọng số tối ưu (SLSQP)", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    processed_data['weighted_return'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights))
    )
    portfolio_daily_return = processed_data.groupby('time')['weighted_return'].sum().reset_index()
    portfolio_daily_return.rename(columns={'weighted_return': 'daily_return'}, inplace=True)
    portfolio_daily_return['cumulative_portfolio_return'] = (1 + portfolio_daily_return['daily_return']).cumprod()
    
    st.subheader("Lợi nhuận tích lũy của danh mục (SLSQP)")
    st.line_chart(portfolio_daily_return.set_index('time')['cumulative_portfolio_return'])

###########################################
# Trang 3: Portfolio Optimization (SGD)
###########################################
elif page == "Portfolio Optimization (SGD)":
    st.header("Tối ưu hóa danh mục đầu tư (SGD)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý từ file 'processed_stock_data.csv'.")
    except Exception:
        st.error("Không tìm thấy file 'processed_stock_data.csv'. Vui lòng chuyển đến trang 'Fetch Stock Data' để tải dữ liệu.")
        st.stop()
    
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    pivot_returns = processed_data.pivot(index='time', columns='symbol', values='daily_return')
    cov_matrix = pivot_returns.cov()
    
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def return_based_weights(expected_returns):
        total_returns = np.sum(expected_returns)
        return expected_returns / total_returns
    
    weights = return_based_weights(expected_returns)
    
    st.subheader("Initial Portfolio Weights (dựa trên kỳ vọng lợi nhuận):")
    for i, symbol in enumerate(expected_returns.index):
        st.write(f"Stock: {symbol}, Weight: {weights[i]:.4f}")
    
    def sgd_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000):
        weights = return_based_weights(expected_returns)
        for epoch in range(epochs):
            grad = np.dot(cov_matrix, weights) / portfolio_volatility(weights, cov_matrix)
            weights -= learning_rate * grad
            weights = np.maximum(weights, 0)
            weights /= np.sum(weights)
            if epoch % 100 == 0:
                st.write(f"Epoch {epoch + 1}, Volatility: {portfolio_volatility(weights, cov_matrix):.6f}")
        return weights
    
    optimal_weights_sgd = sgd_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000)
    
    st.subheader("Optimal Portfolio Weights (SGD):")
    for i, symbol in enumerate(expected_returns.index):
        st.write(f"Stock: {symbol}, Optimal Weight: {optimal_weights_sgd[i]:.4f}")
    
    portfolio_data_sgd = pd.DataFrame({
        'Stock': expected_returns.index,
        'Optimal Weight': optimal_weights_sgd
    })
    
    fig_sgd = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Optimal Portfolio Weights (Pie)', 'Optimal Portfolio Weights (Bar)'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )
    fig_sgd.add_trace(
        go.Pie(
            labels=portfolio_data_sgd['Stock'],
            values=portfolio_data_sgd['Optimal Weight'],
            hole=0.3,
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(
                colors=['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2', '#FF1493', '#00FA9A'],
                line=dict(color='#000000', width=2)
            ),
            hoverinfo='label+percent',
        ),
        row=1, col=1
    )
    fig_sgd.add_trace(
        go.Bar(
            x=portfolio_data_sgd['Stock'],
            y=portfolio_data_sgd['Optimal Weight'],
            marker=dict(
                color=['#FFA07A', '#7B68EE', '#98FB98', '#D2691E', '#6495ED', '#FF69B4', '#2E8B57'],
                line=dict(color='#000000', width=2)
            ),
        ),
        row=1, col=2
    )
    fig_sgd.update_layout(
        title='Optimal Portfolio Weights (SGD) Comparison',
        title_x=0.5,
        height=500,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    st.plotly_chart(fig_sgd, use_container_width=True)
    
    processed_data['weighted_return_sgd'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_sgd))
    )
    portfolio_daily_return_sgd = processed_data.groupby('time')['weighted_return_sgd'].sum().reset_index()
    portfolio_daily_return_sgd.rename(columns={'weighted_return_sgd': 'daily_return'}, inplace=True)
    portfolio_daily_return_sgd['cumulative_portfolio_return'] = (1 + portfolio_daily_return_sgd['daily_return']).cumprod()
    
    st.subheader("Lợi nhuận tích lũy của danh mục (SGD)")
    st.line_chart(portfolio_daily_return_sgd.set_index('time')['cumulative_portfolio_return'])

###########################################
# Trang 4: Portfolio Optimization (SGD - Sharpe)
###########################################
elif page == "Portfolio Optimization (SGD - Sharpe)":
    st.header("Portfolio Optimization Using SGD with Sharpe Ratio Maximization")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý từ file 'processed_stock_data.csv'.")
    except Exception:
        st.error("Không tìm thấy file 'processed_stock_data.csv'. Vui lòng chuyển đến trang 'Fetch Stock Data' để tải dữ liệu.")
        st.stop()
    
    if 'symbol' not in processed_data.columns or 'daily_return' not in processed_data.columns:
        st.error("Processed data must include 'symbol' and 'daily_return' columns.")
        st.stop()
    if processed_data.isnull().any().any():
        st.error("Input data contains null values. Please clean the data before proceeding.")
        st.stop()
    
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    cov_matrix = processed_data.pivot(index='time', columns='symbol', values='daily_return').cov()
    
    def sgd_portfolio_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=2000, tolerance=1e-6):
        weights = expected_returns / expected_returns.sum()
        weights = weights.values
        previous_weights = weights.copy()
        best_sharpe_ratio = -np.inf
        best_weights = weights.copy()
        for epoch in range(epochs):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
            if sharpe_ratio > best_sharpe_ratio:
                best_sharpe_ratio = sharpe_ratio
                best_weights = weights.copy()
            grad = 2 * np.dot(cov_matrix, weights)
            weights -= learning_rate * grad
            weights = np.maximum(weights, 0)
            weights /= np.sum(weights)
            if np.allclose(weights, previous_weights, atol=tolerance):
                st.write(f"Convergence reached after {epoch + 1} epochs")
                break
            previous_weights = weights.copy()
            if epoch % 100 == 0:
                st.write(f"Epoch {epoch + 1}, Sharpe Ratio: {sharpe_ratio:.4f}")
        return best_weights, best_sharpe_ratio
    
    optimal_weights_sgd_bsharp, best_sharpe_ratio = sgd_portfolio_optimization(expected_returns, cov_matrix)
    
    st.subheader("Optimal Portfolio Weights (SGD - Sharpe):")
    for i, symbol in enumerate(expected_returns.index):
        st.write(f"Stock: {symbol}, Optimal Weight: {optimal_weights_sgd_bsharp[i]:.4f}")
    st.write(f"Best Sharpe Ratio: {best_sharpe_ratio:.4f}")
    
    portfolio_data = pd.DataFrame({
        'Stock': expected_returns.index,
        'Optimal Weight': optimal_weights_sgd_bsharp
    })
    
    fig_sharpe = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Optimal Portfolio Weights (Pie)', 'Optimal Portfolio Weights (Bar)'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )
    fig_sharpe.add_trace(
        go.Pie(
            labels=portfolio_data['Stock'],
            values=portfolio_data['Optimal Weight'],
            hole=0.3,
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(
                colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15C', '#19D895', '#F2A900'],
                line=dict(color='#000000', width=2)
            ),
            hoverinfo='label+percent',
        ),
        row=1, col=1
    )
    fig_sharpe.add_trace(
        go.Bar(
            x=portfolio_data['Stock'],
            y=portfolio_data['Optimal Weight'],
            marker=dict(
                color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15C', '#19D895', '#F2A900'],
                line=dict(color='#000000', width=2)
            ),
        ),
        row=1, col=2
    )
    fig_sharpe.update_layout(
        title='Optimal Portfolio Weights (SGD - Sharpe) Comparison',
        title_x=0.5,
        title_font=dict(size=18, family='Arial', color='#1f77b4'),
        height=500,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

    # Thêm phần Lợi nhuận tích lũy của danh mục (SGD - Sharpe)
    processed_data['weighted_return_sharpe'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_sgd_bsharp))
    )
    portfolio_daily_return_sharpe = processed_data.groupby('time')['weighted_return_sharpe'].sum().reset_index()
    portfolio_daily_return_sharpe.rename(columns={'weighted_return_sharpe': 'daily_return'}, inplace=True)
    portfolio_daily_return_sharpe['cumulative_portfolio_return'] = (1 + portfolio_daily_return_sharpe['daily_return']).cumprod()
    
    st.subheader("Lợi nhuận tích lũy của danh mục (SGD - Sharpe)")
    st.line_chart(portfolio_daily_return_sharpe.set_index('time')['cumulative_portfolio_return'])

###########################################
# Trang 5: Comparison with VN-Index (SLSQP)
###########################################
elif page == "Comparison with VN-Index (SLSQP)":
    st.header("So sánh danh mục đầu tư với VN-Index (SLSQP)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý thành công.")
    except FileNotFoundError:
        st.error("File 'processed_stock_data.csv' không tồn tại. Vui lòng Fetch Stock Data trước.")
        st.stop()
    
    # Tối ưu danh mục bằng SLSQP
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    pivot_returns = processed_data.pivot(index='time', columns='symbol', values='daily_return')
    cov_matrix = pivot_returns.cov()
    
    def objective(weights, expected_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(len(expected_returns)))
    total_expected_return = expected_returns.sum()
    init_weights = expected_returns / total_expected_return
    result = minimize(objective, init_weights, args=(expected_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights_slsqp = result.x
    
    # Tính lợi nhuận danh mục
    processed_data['weighted_return_slsqp'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_slsqp))
    )
    portfolio_daily_return_slsqp = processed_data.groupby('time')['weighted_return_slsqp'].sum().reset_index()
    portfolio_daily_return_slsqp.rename(columns={'weighted_return_slsqp': 'daily_return'}, inplace=True)
    portfolio_daily_return_slsqp['cumulative_portfolio_return'] = (1 + portfolio_daily_return_slsqp['daily_return']).cumprod()
    
    # Tải VN-Index
    try:
        vnindex_data = pd.read_csv("vnindex_data.csv")
        vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
        st.success("Đã tải dữ liệu VN-Index từ file 'vnindex_data.csv'.")
    except:
        st.warning("Không tìm thấy file 'vnindex_data.csv'. Đang tải dữ liệu VN-Index...")
        try:
            stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
            vnindex_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
            vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
            vnindex_data.to_csv("vnindex_data.csv", index=False)
            st.success("Đã lưu dữ liệu VN-Index vào file 'vnindex_data.csv'.")
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu VN-Index: {e}")
            st.stop()
    
    # Tính lợi nhuận VN-Index
    vnindex_data['market_return'] = vnindex_data['close'].pct_change()
    vnindex_data['cumulative_daily_return'] = (1 + vnindex_data['market_return']).cumprod()
    
    # Gộp dữ liệu
    portfolio_daily_return_slsqp['time'] = pd.to_datetime(portfolio_daily_return_slsqp['time'])
    comparison_slsqp = pd.merge(portfolio_daily_return_slsqp, vnindex_data[['time', 'cumulative_daily_return']],
                                on='time', how='inner')
    comparison_slsqp.rename(columns={
        'cumulative_portfolio_return': 'Portfolio Return (SLSQP)',
        'cumulative_daily_return': 'VN-Index Return'
    }, inplace=True)
    
    st.subheader("Bảng so sánh lợi nhuận (10 dòng cuối) - SLSQP vs VN-Index")
    st.dataframe(comparison_slsqp[['time', 'Portfolio Return (SLSQP)', 'VN-Index Return']].tail(10))
    
    fig_comp_slsqp = go.Figure()
    fig_comp_slsqp.add_trace(go.Scatter(
        x=comparison_slsqp['time'],
        y=comparison_slsqp['Portfolio Return (SLSQP)'],
        mode='lines',
        name='Portfolio Return (SLSQP)',
        line=dict(color='blue', width=2),
        hovertemplate='Date: %{x}<br>Portfolio Return (SLSQP): %{y:.2%}<extra></extra>'
    ))
    fig_comp_slsqp.add_trace(go.Scatter(
        x=comparison_slsqp['time'],
        y=comparison_slsqp['VN-Index Return'],
        mode='lines',
        name='VN-Index Return',
        line=dict(color='red', width=2),
        hovertemplate='Date: %{x}<br>VN-Index Return: %{y:.2%}<extra></extra>'
    ))
    fig_comp_slsqp.update_layout(
        title="So sánh lợi nhuận danh mục (SLSQP) vs VN-Index",
        xaxis_title="Thời gian",
        yaxis_title="Lợi nhuận tích lũy",
        template="plotly_white"
    )
    st.plotly_chart(fig_comp_slsqp, use_container_width=True)
    
    comparison_slsqp.to_csv("portfolio_vs_vnindex_comparison_slsqp.csv", index=False)
    st.write("Comparison saved to 'portfolio_vs_vnindex_comparison_slsqp.csv'.")

###########################################
# Trang 6: Comparison with VN-Index (SGD)
###########################################
elif page == "Comparison with VN-Index (SGD)":
    st.header("So sánh danh mục đầu tư với VN-Index (SGD)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý thành công.")
    except FileNotFoundError:
        st.error("File 'processed_stock_data.csv' không tồn tại. Vui lòng Fetch Stock Data trước.")
        st.stop()
    
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    pivot_returns = processed_data.pivot(index='time', columns='symbol', values='daily_return')
    cov_matrix = pivot_returns.cov()
    
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def return_based_weights(expected_returns):
        return expected_returns / expected_returns.sum()
    
    def sgd_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000):
        weights = return_based_weights(expected_returns)
        for epoch in range(epochs):
            grad = np.dot(cov_matrix, weights) / portfolio_volatility(weights, cov_matrix)
            weights -= learning_rate * grad
            weights = np.maximum(weights, 0)
            weights /= np.sum(weights)
        return weights
    
    optimal_weights_sgd = sgd_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000)
    
    processed_data['weighted_return_sgd'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_sgd))
    )
    portfolio_daily_return_sgd = processed_data.groupby('time')['weighted_return_sgd'].sum().reset_index()
    portfolio_daily_return_sgd.rename(columns={'weighted_return_sgd': 'daily_return'}, inplace=True)
    portfolio_daily_return_sgd['cumulative_portfolio_return'] = (1 + portfolio_daily_return_sgd['daily_return']).cumprod()
    
    try:
        vnindex_data = pd.read_csv("vnindex_data.csv")
        vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
        st.success("VN-Index data loaded successfully from 'vnindex_data.csv'.")
    except:
        st.warning("Không tìm thấy file 'vnindex_data.csv'. Đang tải dữ liệu VN-Index...")
        try:
            stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
            vnindex_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
            vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
            vnindex_data.to_csv("vnindex_data.csv", index=False)
            st.success("VN-Index data saved successfully.")
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu VN-Index: {e}")
            st.stop()
    
    vnindex_data['market_return'] = vnindex_data['close'].pct_change()
    vnindex_data['cumulative_daily_return'] = (1 + vnindex_data['market_return']).cumprod()
    
    portfolio_daily_return_sgd['time'] = pd.to_datetime(portfolio_daily_return_sgd['time'])
    comparison_sgd = pd.merge(
        portfolio_daily_return_sgd,
        vnindex_data[['time', 'cumulative_daily_return']],
        on='time',
        how='inner'
    )
    comparison_sgd.rename(columns={
        'cumulative_portfolio_return': 'Portfolio Return (SGD)',
        'cumulative_daily_return': 'VN-Index Return'
    }, inplace=True)
    
    st.subheader("Bảng so sánh lợi nhuận (SGD) vs VN-Index (10 dòng cuối)")
    st.dataframe(comparison_sgd[['time', 'Portfolio Return (SGD)', 'VN-Index Return']].tail(10))
    
    fig_comp_sgd = go.Figure()
    fig_comp_sgd.add_trace(go.Scatter(
        x=comparison_sgd['time'],
        y=comparison_sgd['Portfolio Return (SGD)'],
        mode='lines',
        name='Portfolio Return (SGD)',
        line=dict(color='green', width=2),
        hovertemplate='Date: %{x}<br>Portfolio Return (SGD): %{y:.2%}<extra></extra>'
    ))
    fig_comp_sgd.add_trace(go.Scatter(
        x=comparison_sgd['time'],
        y=comparison_sgd['VN-Index Return'],
        mode='lines',
        name='VN-Index Return',
        line=dict(color='red', width=2),
        hovertemplate='Date: %{x}<br>VN-Index Return: %{y:.2%}<extra></extra>'
    ))
    fig_comp_sgd.update_layout(
        title="Comparison of Portfolio Return (SGD) vs VN-Index Return",
        xaxis_title="Time",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig_comp_sgd, use_container_width=True)
    comparison_sgd.to_csv("portfolio_vs_vnindex_comparison_sgd.csv", index=False)
    st.write("Comparison saved to 'portfolio_vs_vnindex_comparison_sgd.csv'.")

###########################################
# Trang 7: Comparison with VN-Index (SGD - Sharpe)
###########################################
elif page == "Comparison with VN-Index (SGD - Sharpe)":
    st.header("Comparison of Portfolio (SGD - Sharpe) vs. VN-Index Returns")
    
    # Tải dữ liệu đã xử lý
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Processed data loaded successfully.")
    except Exception:
        st.error("File 'processed_stock_data.csv' not found. Please go to 'Fetch Stock Data' page to load data.")
        st.stop()
    
    # Tính toán kỳ vọng lợi nhuận và ma trận hiệp phương sai
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    cov_matrix = processed_data.pivot(index='time', columns='symbol', values='daily_return').cov()
    
    # Hàm tối ưu hóa SGD cho Sharpe Ratio
    def sgd_portfolio_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=2000, tolerance=1e-6):
        weights = expected_returns / expected_returns.sum()
        weights = weights.values
        previous_weights = weights.copy()
        best_sharpe_ratio = -np.inf
        best_weights = weights.copy()
        for epoch in range(epochs):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
            if sharpe_ratio > best_sharpe_ratio:
                best_sharpe_ratio = sharpe_ratio
                best_weights = weights.copy()
            grad = 2 * np.dot(cov_matrix, weights)
            weights -= learning_rate * grad
            weights = np.maximum(weights, 0)
            weights /= np.sum(weights)
            if np.allclose(weights, previous_weights, atol=tolerance):
                st.write(f"Convergence reached after {epoch + 1} epochs")
                break
            previous_weights = weights.copy()
        return best_weights
    
    # Tính toán trọng số tối ưu
    optimal_weights_sgd_bsharp = sgd_portfolio_optimization(expected_returns, cov_matrix)
    
    # Tính toán lợi nhuận danh mục
    processed_data['weighted_return_sharpe'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_sgd_bsharp))
    )
    portfolio_daily_return_sharpe = processed_data.groupby('time')['weighted_return_sharpe'].sum().reset_index()
    portfolio_daily_return_sharpe.rename(columns={'weighted_return_sharpe': 'daily_return'}, inplace=True)
    portfolio_daily_return_sharpe['cumulative_portfolio_return'] = (1 + portfolio_daily_return_sharpe['daily_return']).cumprod()
    portfolio_daily_return_sharpe['time'] = pd.to_datetime(portfolio_daily_return_sharpe['time'])
    
    # Tải dữ liệu VN-Index
    try:
        vnindex_data = pd.read_csv("vnindex_data.csv")
        vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
        st.success("VN-Index data loaded successfully.")
    except:
        st.warning("VN-Index data not found. Attempting to load it...")
        try:
            stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
            vnindex_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
            vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
            vnindex_data.to_csv("vnindex_data.csv", index=False)
            st.success("VN-Index data saved successfully.")
        except Exception as e:
            st.error(f"Error loading VN-Index data: {e}")
            st.stop()
    
    # Tính toán lợi nhuận tích lũy của VN-Index
    vnindex_data['market_return'] = vnindex_data['close'].pct_change()
    vnindex_data['cumulative_daily_return'] = (1 + vnindex_data['market_return']).cumprod()
    
    # Gộp dữ liệu danh mục và VN-Index
    comparison_sharpe = pd.merge(
        portfolio_daily_return_sharpe,
        vnindex_data[['time', 'cumulative_daily_return']],
        on='time',
        how='inner'
    )
    comparison_sharpe.rename(columns={
        'cumulative_portfolio_return': 'Portfolio Return (Sharpe)',
        'cumulative_daily_return': 'VN-Index Return'
    }, inplace=True)
    
    # Hiển thị bảng so sánh (10 dòng cuối)
    st.subheader("Comparison Table (Last 10 rows)")
    st.dataframe(comparison_sharpe[['time', 'Portfolio Return (Sharpe)', 'VN-Index Return']].tail(10))
    
    # Vẽ biểu đồ so sánh
    fig_comp_sharpe = go.Figure()
    fig_comp_sharpe.add_trace(go.Scatter(
        x=comparison_sharpe['time'],
        y=comparison_sharpe['Portfolio Return (Sharpe)'],
        mode='lines',
        name='Portfolio Return (Sharpe)',
        line=dict(color='orange', width=2),
        hovertemplate='Date: %{x}<br>Portfolio Return (Sharpe): %{y:.2%}<extra></extra>'
    ))
    fig_comp_sharpe.add_trace(go.Scatter(
        x=comparison_sharpe['time'],
        y=comparison_sharpe['VN-Index Return'],
        mode='lines',
        name='VN-Index Return',
        line=dict(color='red', width=2),
        hovertemplate='Date: %{x}<br>VN-Index Return: %{y:.2%}<extra></extra>'
    ))
    fig_comp_sharpe.update_layout(
        title="Comparison of Portfolio Return (Sharpe Optimization) vs VN-Index Return",
        xaxis_title="Time",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig_comp_sharpe, use_container_width=True)
    
    # Lưu kết quả so sánh vào file CSV
    comparison_sharpe.to_csv("portfolio_vs_vnindex_comparison_sharpe.csv", index=False)
    st.write("Comparison saved to 'portfolio_vs_vnindex_comparison_sharpe.csv'.")

###########################################
# Trang 8: Data Visualization
###########################################
elif page == "Data Visualization":
    st.header("Data Visualization")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
    except Exception as e:
        st.error("Không thể tải file 'processed_stock_data.csv'. Vui lòng chuyển đến trang 'Fetch Stock Data' để tải dữ liệu.")
        st.stop()
    
    st.subheader("Stock Closing Price Trend Over Time")
    fig1 = px.line(
        processed_data,
        x='time',
        y='close',
        color='symbol',
        title='Stock Closing Price Trend Over Time',
        labels={'time': 'Time', 'close': 'Closing Price', 'symbol': 'Stock Symbol'},
    )
    fig1.update_layout(
        xaxis_title='Time',
        yaxis_title='Closing Price',
        legend_title='Stock Symbol',
        template='plotly_white',
        hovermode='x unified',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Correlation Heatmap of Closing Prices")
    close_data = processed_data.pivot_table(values='close', index='time', columns='symbol')
    correlation_matrix = close_data.corr()
    rounded_correlation = correlation_matrix.round(2)
    fig2 = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        colorbar=dict(title='Correlation Coefficient'),
    ))
    for i in range(len(rounded_correlation)):
        for j in range(len(rounded_correlation.columns)):
            fig2.add_annotation(
                text=str(rounded_correlation.iloc[i, j]),
                x=rounded_correlation.columns[j],
                y=rounded_correlation.index[i],
                showarrow=False,
                font=dict(color='black' if rounded_correlation.iloc[i, j] < 0 else 'white')
            )
    fig2.update_traces(
        hovertemplate='<b>Stock Symbol: %{x}</b><br>' +
                      '<b>Stock Symbol: %{y}</b><br>' +
                      'Correlation Coefficient: %{z:.4f}<extra></extra>'
    )
    fig2.update_layout(
        title='Correlation Heatmap of Closing Prices',
        xaxis_title='Stock Symbol',
        yaxis_title='Stock Symbol'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Heat Map of Daily Returns")
    returns_data = processed_data.pivot_table(index='time', columns='symbol', values='daily_return')
    correlation_matrix_returns = returns_data.corr()
    fig3 = ff.create_annotated_heatmap(
        z=correlation_matrix_returns.values,
        x=correlation_matrix_returns.columns.tolist(),
        y=correlation_matrix_returns.columns.tolist(),
        colorscale='RdBu',
        zmin=-1, zmax=1
    )
    fig3.update_layout(title="Correlation Matrix Between Stocks")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Stock Volatility Over Time")
    fig4 = px.line(processed_data, x='time', y='volatility', color='symbol', title="Stock Volatility Over Time")
    fig4.update_xaxes(title_text='Date')
    fig4.update_yaxes(title_text='Volatility')
    st.plotly_chart(fig4, use_container_width=True)

###########################################
# Trang 9: Company Information
###########################################
elif page == "Company Information":
    st.header("Thông tin tổng hợp về các công ty")
    
    if 'symbols' not in st.session_state:
        st.error("Vui lòng nhập mã cổ phiếu ở trang 'Fetch Stock Data' trước.")
    else:
        symbols = st.session_state['symbols']
        
        for symbol in symbols:
            st.subheader(f"Thông tin cho mã {symbol}")
            try:
                company = Vnstock().stock(symbol=symbol, source='TCBS').company
                
                # Hồ sơ công ty
                st.write("**Hồ sơ công ty:**")
                profile = company.profile()
                if isinstance(profile, pd.DataFrame):
                    st.dataframe(profile)
                else:
                    st.write(profile)  # Display raw output if not a DataFrame
                
                # Cổ đông
                st.write("**Cổ đông:**")
                shareholders = company.shareholders()
                if isinstance(shareholders, pd.DataFrame):
                    st.dataframe(shareholders)
                else:
                    st.write(shareholders)
                
                # Giao dịch nội bộ
                st.write("**Giao dịch nội bộ:**")
                insider_deals = company.insider_deals()
                if isinstance(insider_deals, pd.DataFrame):
                    st.dataframe(insider_deals)
                else:
                    st.write(insider_deals)
                
                # Công ty con
                st.write("**Công ty con:**")
                subsidiaries = company.subsidiaries()
                if isinstance(subsidiaries, pd.DataFrame):
                    st.dataframe(subsidiaries)
                else:
                    st.write(subsidiaries)
                
                # Ban điều hành
                st.write("**Ban điều hành:**")
                officers = company.officers()
                if isinstance(officers, pd.DataFrame):
                    st.dataframe(officers)
                else:
                    st.write(officers)
                
                # Sự kiện
                st.write("**Sự kiện:**")
                events = company.events()
                if isinstance(events, pd.DataFrame):
                    st.dataframe(events)
                else:
                    st.write(events)
                
                # Tin tức
                st.write("**Tin tức:**")
                news = company.news()
                if isinstance(news, list) and all(isinstance(item, dict) for item in news):
                    for item in news:
                        st.write(f"- {item.get('title', 'N/A')} ({item.get('date', 'N/A')})")
                        st.write(item.get('summary', 'No summary available'))
                        url = item.get('url', None)
                        if url:
                            st.write(f"[Đọc thêm]({url})")
                        else:
                            st.write("No URL available")
                else:
                    st.write("Tin tức không khả dụng hoặc định dạng không đúng:")
                    st.write(news)  # Display raw output for debugging
                
                # Cổ tức
                st.write("**Cổ tức:**")
                dividends = company.dividends()
                if isinstance(dividends, pd.DataFrame):
                    st.dataframe(dividends)
                else:
                    st.write(dividends)
            
            except Exception as e:
                st.error(f"Lỗi khi tải thông tin cho mã {symbol}: {e}")
###########################################
# Trang 10: Financial Statements
###########################################
elif page == "Financial Statements":
    st.header("Tổng hợp báo cáo tài chính")

    # Cấu hình Plotly: modebar luôn hiển thị
    config = {
        "displayModeBar": True,
        "displaylogo": False
    }
    
    if 'symbols' not in st.session_state:
        st.error("Vui lòng nhập mã cổ phiếu ở trang 'Fetch Stock Data' trước.")
    else:
        symbols = st.session_state['symbols']

        def rename_duplicate_columns(df):
            if df.empty:
                return df
            if isinstance(df.columns, pd.MultiIndex):
                flat_columns = [
                    '_'.join(str(col).strip() for col in multi_col if str(col).strip())
                    for multi_col in df.columns
                ]
            else:
                flat_columns = df.columns.tolist()
            seen = {}
            final_columns = []
            for col in flat_columns:
                if col in seen:
                    seen[col] += 1
                    final_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    final_columns.append(col)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.Index(final_columns)
            else:
                df.columns = final_columns
            return df

        # CSS cho nội dung của expander (background trắng, đổ bóng,...)
        st.markdown(
            """
            <style>
            .streamlit-expanderContent {
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        for symbol in symbols:
            st.header(f"Báo cáo tài chính cho mã {symbol}")

            # ----------------------- BALANCE SHEET -----------------------
            with st.expander("Bảng cân đối kế toán (Hàng năm)"):
                try:
                    stock = Vnstock().stock(symbol=symbol, source='VCI')
                    balance_data = stock.finance.balance_sheet(period='year', lang='vi', dropna=True)
                    balance_data = rename_duplicate_columns(balance_data)
                    if not balance_data.empty and 'Năm' in balance_data.columns:
                        st.write("**Bảng cân đối kế toán (Hàng năm):**")
                        st.dataframe(balance_data)
                        numeric_cols = [col for col in balance_data.select_dtypes(include=['float64', 'int64']).columns if col != 'Năm']
                        if numeric_cols:
                            selected_cols = st.multiselect(
                                f"Chọn các chỉ số để hiển thị biểu đồ (Bảng cân đối {symbol}):",
                                options=numeric_cols,
                                default=[]
                            )
                            available_years = sorted(balance_data['Năm'].unique())
                            selected_years = st.multiselect(
                                f"Chọn năm hiển thị cho biểu đồ (Bảng cân đối {symbol}):",
                                options=available_years,
                                default=[]
                            )
                            df_filtered = balance_data[balance_data['Năm'].isin(selected_years)] if selected_years else balance_data

                            if selected_cols:
                                for i in range(0, len(selected_cols), 5):
                                    cols = st.columns(5)
                                    for j, col in enumerate(selected_cols[i:i+5]):
                                        with cols[j]:
                                            st.markdown(f"**{col}**")
                                            tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ đường"])
                                            
                                            with tab1:
                                                bar_fig = go.Figure()
                                                bar_fig.add_trace(go.Bar(
                                                    x=df_filtered['Năm'],
                                                    y=df_filtered[col],
                                                    name=col,
                                                    marker_color=random_color(),
                                                    hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                                ))
                                                bar_fig.update_layout(
                                                    title=f"{col} - {symbol}",
                                                    xaxis_title="Năm",
                                                    yaxis_title="Giá trị (Tỷ đồng)",
                                                    template="plotly_white",
                                                    height=300,
                                                    margin=dict(l=20, r=20, t=150, b=20)
                                                )
                                                st.plotly_chart(bar_fig, use_container_width=True, config=config, key=f"balance_{symbol}_{col}_bar")
                                            
                                            with tab2:
                                                line_fig = go.Figure()
                                                line_fig.add_trace(go.Scatter(
                                                    x=df_filtered['Năm'],
                                                    y=df_filtered[col],
                                                    mode='lines+markers',
                                                    marker_color=random_color(),
                                                    hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                                ))
                                                line_fig.update_layout(
                                                    title=f"{col} - {symbol} (Đường)",
                                                    xaxis_title="Năm",
                                                    yaxis_title="Giá trị (Tỷ đồng)",
                                                    template="plotly_white",
                                                    height=300,
                                                    margin=dict(l=20, r=20, t=150, b=20)
                                                )
                                                st.plotly_chart(line_fig, use_container_width=True, config=config, key=f"balance_{symbol}_{col}_line")
                    else:
                        st.warning(f"Không có dữ liệu hoặc cột 'Năm' cho bảng cân đối kế toán của {symbol}")
                except Exception as e:
                    st.error(f"Lỗi khi tải bảng cân đối kế toán cho mã {symbol}: {e}")

            # ----------------------- INCOME STATEMENT -----------------------
            with st.expander("Báo cáo lãi lỗ (Hàng năm)"):
                try:
                    stock = Vnstock().stock(symbol=symbol, source='VCI')
                    income_data = stock.finance.income_statement(period='year', lang='vi', dropna=True)
                    income_data = rename_duplicate_columns(income_data)
                    if not income_data.empty and 'Năm' in income_data.columns:
                        st.write("**Báo cáo lãi lỗ (Hàng năm):**")
                        st.dataframe(income_data)
                        numeric_cols = [col for col in income_data.select_dtypes(include=['float64', 'int64']).columns if col != 'Năm']
                        if numeric_cols:
                            selected_cols = st.multiselect(
                                f"Chọn các chỉ số để hiển thị biểu đồ (Báo cáo lãi lỗ {symbol}):",
                                options=numeric_cols,
                                default=[]
                            )
                            available_years = sorted(income_data['Năm'].unique())
                            selected_years = st.multiselect(
                                f"Chọn năm hiển thị cho biểu đồ (Báo cáo lãi lỗ {symbol}):",
                                options=available_years,
                                default=[]
                            )
                            df_filtered = income_data[income_data['Năm'].isin(selected_years)] if selected_years else income_data

                            if selected_cols:
                                for i in range(0, len(selected_cols), 5):
                                    cols = st.columns(5)
                                    for j, col in enumerate(selected_cols[i:i+5]):
                                        with cols[j]:
                                            st.markdown(f"**{col}**")
                                            tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ đường"])
                                            
                                            with tab1:
                                                bar_fig = go.Figure()
                                                bar_fig.add_trace(go.Bar(
                                                    x=df_filtered['Năm'],
                                                    y=df_filtered[col],
                                                    name=col,
                                                    marker_color=random_color(),
                                                    hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                                ))
                                                bar_fig.update_layout(
                                                    title=f"{col} - {symbol}",
                                                    xaxis_title="Năm",
                                                    yaxis_title="Giá trị (Tỷ đồng)",
                                                    template="plotly_white",
                                                    height=300,
                                                    margin=dict(l=20, r=20, t=150, b=20)
                                                )
                                                st.plotly_chart(bar_fig, use_container_width=True, config=config, key=f"income_{symbol}_{col}_bar")
                                            
                                            with tab2:
                                                line_fig = go.Figure()
                                                line_fig.add_trace(go.Scatter(
                                                    x=df_filtered['Năm'],
                                                    y=df_filtered[col],
                                                    mode='lines+markers',
                                                    marker_color=random_color(),
                                                    hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                                ))
                                                line_fig.update_layout(
                                                    title=f"{col} - {symbol} (Đường)",
                                                    xaxis_title="Năm",
                                                    yaxis_title="Giá trị (Tỷ đồng)",
                                                    template="plotly_white",
                                                    height=300,
                                                    margin=dict(l=20, r=20, t=150, b=20)
                                                )
                                                st.plotly_chart(line_fig, use_container_width=True, config=config, key=f"income_{symbol}_{col}_line")
                    else:
                        st.warning(f"Không có dữ liệu hoặc cột 'Năm' cho báo cáo lãi lỗ của {symbol}")
                except Exception as e:
                    st.error(f"Lỗi khi tải báo cáo lãi lỗ cho mã {symbol}: {e}")

            # ----------------------- CASH FLOW -----------------------
            with st.expander("Báo cáo lưu chuyển tiền tệ (Hàng năm)"):
                try:
                    stock = Vnstock().stock(symbol=symbol, source='VCI')
                    cash_flow_data = stock.finance.cash_flow(period='year', lang="vi", dropna=True)
                    cash_flow_data = rename_duplicate_columns(cash_flow_data)
                    if not cash_flow_data.empty and 'Năm' in cash_flow_data.columns:
                        st.write("**Báo cáo lưu chuyển tiền tệ (Hàng năm):**")
                        st.dataframe(cash_flow_data)
                        numeric_cols = [col for col in cash_flow_data.select_dtypes(include=['float64', 'int64']).columns if col != 'Năm']
                        if numeric_cols:
                            selected_cols = st.multiselect(
                                f"Chọn các chỉ số để hiển thị biểu đồ (Báo cáo lưu chuyển {symbol}):",
                                options=numeric_cols,
                                default=[]
                            )
                            available_years = sorted(cash_flow_data['Năm'].unique())
                            selected_years = st.multiselect(
                                f"Chọn năm hiển thị cho biểu đồ (Báo cáo lưu chuyển {symbol}):",
                                options=available_years,
                                default=[]
                            )
                            df_filtered = cash_flow_data[cash_flow_data['Năm'].isin(selected_years)] if selected_years else cash_flow_data

                            if selected_cols:
                                for i in range(0, len(selected_cols), 5):
                                    cols = st.columns(5)
                                    for j, col in enumerate(selected_cols[i:i+5]):
                                        with cols[j]:
                                            st.markdown(f"**{col}**")
                                            tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ đường"])
                                            
                                            with tab1:
                                                bar_fig = go.Figure()
                                                bar_fig.add_trace(go.Bar(
                                                    x=df_filtered['Năm'],
                                                    y=df_filtered[col],
                                                    name=col,
                                                    marker_color=random_color(),
                                                    hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                                ))
                                                bar_fig.update_layout(
                                                    title=f"{col} - {symbol}",
                                                    xaxis_title="Năm",
                                                    yaxis_title="Giá trị (Tỷ đồng)",
                                                    template="plotly_white",
                                                    height=300,
                                                    margin=dict(l=20, r=20, t=150, b=20)
                                                )
                                                st.plotly_chart(bar_fig, use_container_width=True, config=config, key=f"cashflow_{symbol}_{col}_bar")
                                            
                                            with tab2:
                                                line_fig = go.Figure()
                                                line_fig.add_trace(go.Scatter(
                                                    x=df_filtered['Năm'],
                                                    y=df_filtered[col],
                                                    mode='lines+markers',
                                                    marker_color=random_color(),
                                                    hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                                ))
                                                line_fig.update_layout(
                                                    title=f"{col} - {symbol} (Đường)",
                                                    xaxis_title="Năm",
                                                    yaxis_title="Giá trị (Tỷ đồng)",
                                                    template="plotly_white",
                                                    height=300,
                                                    margin=dict(l=20, r=20, t=150, b=20)
                                                )
                                                st.plotly_chart(line_fig, use_container_width=True, config=config, key=f"cashflow_{symbol}_{col}_line")
                    else:
                        st.warning(f"Không có dữ liệu hoặc cột 'Năm' cho báo cáo lưu chuyển tiền tệ của {symbol}")
                except Exception as e:
                    st.error(f"Lỗi khi tải báo cáo lưu chuyển tiền tệ cho mã {symbol}: {e}")

            # ----------------------- FINANCIAL RATIOS -----------------------
            with st.expander("Chỉ số tài chính (Hàng năm)"):
                try:
                    stock = Vnstock().stock(symbol=symbol, source='VCI')
                    ratios_data = stock.finance.ratio(period='year', lang='vi', dropna=True)
                    ratios_data = rename_duplicate_columns(ratios_data)
                    if not ratios_data.empty and 'Meta_Năm' in ratios_data.columns:
                        st.write("**Chỉ số tài chính (Hàng năm):**")
                        st.dataframe(ratios_data)
                        numeric_cols = [col for col in ratios_data.select_dtypes(include=['float64', 'int64']).columns if col != 'Meta_Năm']
                        if numeric_cols:
                            selected_cols = st.multiselect(
                                f"Chọn các chỉ số để hiển thị biểu đồ (Chỉ số tài chính {symbol}):",
                                options=numeric_cols,
                                default=[]
                            )
                            available_years = sorted(ratios_data['Meta_Năm'].unique())
                            selected_years = st.multiselect(
                                f"Chọn năm hiển thị cho biểu đồ (Chỉ số tài chính {symbol}):",
                                options=available_years,
                                default=[]
                            )
                            df_filtered = ratios_data[ratios_data['Meta_Năm'].isin(selected_years)] if selected_years else ratios_data

                            if selected_cols:
                                for i in range(0, len(selected_cols), 5):
                                    cols = st.columns(5)
                                    for j, col in enumerate(selected_cols[i:i+5]):
                                        with cols[j]:
                                            st.markdown(f"**{col}**")
                                            tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ đường"])
                                            
                                            with tab1:
                                                bar_fig = go.Figure()
                                                bar_fig.add_trace(go.Bar(
                                                    x=df_filtered['Meta_Năm'],
                                                    y=df_filtered[col],
                                                    name=col,
                                                    marker_color=random_color(),
                                                    hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                                ))
                                                bar_fig.update_layout(
                                                    title=f"{col} - {symbol}",
                                                    xaxis_title="Năm",
                                                    yaxis_title="Giá trị",
                                                    template="plotly_white",
                                                    height=300,
                                                    margin=dict(l=20, r=20, t=150, b=20)
                                                )
                                                st.plotly_chart(bar_fig, use_container_width=True, config=config, key=f"ratios_{symbol}_{col}_bar")
                                            
                                            with tab2:
                                                line_fig = go.Figure()
                                                line_fig.add_trace(go.Scatter(
                                                    x=df_filtered['Meta_Năm'],
                                                    y=df_filtered[col],
                                                    mode='lines+markers',
                                                    marker_color=random_color(),
                                                    hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                                ))
                                                line_fig.update_layout(
                                                    title=f"{col} - {symbol} (Đường)",
                                                    xaxis_title="Năm",
                                                    yaxis_title="Giá trị",
                                                    template="plotly_white",
                                                    height=300,
                                                    margin=dict(l=20, r=20, t=150, b=20)
                                                )
                                                st.plotly_chart(line_fig, use_container_width=True, config=config, key=f"ratios_{symbol}_{col}_line")
                    else:
                        st.warning(f"Không có dữ liệu hoặc cột 'Meta_Năm' cho chỉ số tài chính của {symbol}")
                except Exception as e:
                    st.error(f"Lỗi khi tải chỉ số tài chính cho mã {symbol}: {e}")
