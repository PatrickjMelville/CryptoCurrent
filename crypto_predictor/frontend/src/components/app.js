import React from 'react';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import Board from './board';
import Navbar from './navbar';
import Predictions from './predictions';



export default function App() {

  return (
    <div className="App">
      <Router>
        <Navbar />
          <Switch>
            <Route exact path='/' component={Board} />
            <Route exact path='/predictions' component={Predictions} />
          </Switch>
      </Router>
    </div>
  );
}