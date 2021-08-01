import React, { useEffect } from 'react';
import { csv } from 'd3';
import data from '../../../predictions.csv'


export default function Predictions() {

  console.log(data);
  
  
  const value = 1200

  return (
    <div className="predictions">
      <div className="predictions-content">
        <p>DISCLAIMER: THIS IS A MACHINE LEARNING EXERCISE NOT FINANCE ADVICE
          <br />
          Predictions are calculated using past price trends/patterns and should not be considered reliable information or recommendations.
        </p>
        <div className="future-prediction">
          <h4>Your selected cryptocurrency is predicted to be ${value} in 14 days from now</h4>
        </div>
        <div className="plot">
          <h1>Predictions Model</h1>
          <p>
            Each predicted price in this model is a prediction one day into the future based on the trends of the previous 60 days.
          </p>
          <img src="../../static/images/plot.png"></img>
        </div>
      </div>
      <img className="robot_predictions" src="../../static/images/robot.png"></img>
    </div>
  );
}