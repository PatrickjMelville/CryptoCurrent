import React from 'react';

const Coin = ({ name, image, symbol, price, volume, priceChange, marketcap }) => {
    return (
        <div className="coin-container">
            <div className="coin-row">
                <div className="coin">
                    <img src={image} alt="cryptocurrency" />
                    <h1>{name}</h1>
                    <p className="symbol">{symbol}</p>
                </div>
                <div className="coin-data">
                    <p className="price">${price}</p>
                    <p className="volume">${volume.toLocaleString()}</p>
                    {priceChange < 0 ? (
                        <p className="percent-change red">{priceChange.toFixed(2)}%</p>
                    ) : (<p className="percent-change green">{priceChange.toFixed(2)}%</p>
                    )}
                    <p className="marketcap">
                        Mkt Cap: ${marketcap.toLocaleString()}
                    </p>
                </div>
            </div>
            
        </div>
    )
}

export default Coin;