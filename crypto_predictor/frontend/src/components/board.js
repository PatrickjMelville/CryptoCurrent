import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Coin from './coin';



export default function Board() {

  const [coins, setCoins] = useState([]);
  const [search, setSearch] = useState('');

  useEffect(() => {
    axios.get('https://api.coingecko.com/api/v3/coins/markets?vs_currency=aud&order=market_cap_desc&per_page=100&page=1&sparkline=false')
    .then(res => {
      setCoins(res.data);
    }).catch(error => alert('API error'));
  }, []);

  const handleChange = e => {
    setSearch(e.target.value);
  }

  const searchedCoins = coins.filter(coin => 
    coin.name.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="Board">
      <div className="main_info">
        <div className="content">
          <div className="title">
            <h1>CRYPTO</h1>
            <h2>CURRENT</h2>
          </div>
          <div className="info">
            <p>Get current Cryptocurrency prices.</p>
            <br />
            <p>Get future price predictions based on Machine Learning.</p>
            <br />
            <div className="search-bar">
              <form>
                <input type="text" placeholder="Search..." className="search-input" onChange={handleChange} />
              </form>
            </div>
          </div>
        </div>
        <img className="robot" src="../../static/images/robot2.png"></img>
      </div>
        {searchedCoins.map(coin => {
            return (
              <Coin 
                key={coin.id} 
                name={coin.name} 
                image={coin.image} 
                symbol={coin.symbol} 
                volume={coin.total_volume} 
                price={coin.current_price} 
                priceChange={coin.price_change_percentage_24h}
                marketcap={coin.market_cap}
              />
            );
          })}
    </div>
  );
}