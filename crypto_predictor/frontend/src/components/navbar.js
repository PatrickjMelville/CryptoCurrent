import React from 'react';
import { Link } from 'react-router-dom';

export default function Navbar() {

  return (
    <>
      <nav className='navbar'>
        <div className='navbar-container'>
          <Link to='/' className='navbar-logo'>
            <h1>CRYPTO</h1>
            <h2>CURRENT</h2>
          </Link>
          <ul className='nav-menu'>
            <li className='nav-item'>
              <Link to='/' className='nav-links'>
                Home
              </Link>
            </li>
            <li className='nav-item'>
              <Link to='/predictions' className='nav-links'>
                Predictions
              </Link>
            </li>
          </ul>
        </div>
      </nav>
    </>
  );
}