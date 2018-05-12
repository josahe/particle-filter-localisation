/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *
 * Modified on: May 11, 2018
 *          By: Joe Herd
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  default_random_engine gen;
  double std_x, std_y, std_theta;

  // Standard deviations for x, y, and theta
  std_x = x + std[0];
  std_y = y + std[1];
  std_theta = theta + std[2];

  // Normal distributions for x, y and theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Number of particles
  num_particles = 1; // TODO raise to 1000

  // Initialize particles to first position estimate (GPS) by randomly drawing
  // from the distribution graphs and by setting weights to 1
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);

    weights.push_back(1.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  double std_velocity, std_yaw_rate;

  // Standard deviations for velocity and yaw rate
  std_velocity = velocity + std_pos[0];
  std_yaw_rate = yaw_rate + std_pos[1];

  // Normal distributions for velocity and yaw rate
  normal_distribution<double> dist_velocity(velocity, std_velocity);
  normal_distribution<double> dist_yaw_rate(yaw_rate, std_yaw_rate);

  // Update particle positions by randomly drawing from the distribution graphs
  for (int i = 0; i < num_particles; ++i) {
    particles[i].x = update_x(particles[i].x, dist_velocity(gen),
      particles[i].theta, dist_yaw_rate(gen), delta_t);
    particles[i].y = update_y(particles[i].y, dist_velocity(gen),
      particles[i].theta, dist_yaw_rate(gen), delta_t);
    particles[i].theta = update_theta(particles[i].theta, dist_yaw_rate(gen),
      delta_t);
  }
}

std::vector<LandmarkObs> ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> mapped, std::vector<LandmarkObs> observations) {
  std::vector<LandmarkObs> corresponding_landmarks;

  // Calculate nearest neighbor (closest measurement in distance)
  for (unsigned int i = 0; i < observations.size(); ++i) {
    LandmarkObs closest_mapped;
    closest_mapped.id = mapped[0].id;
    closest_mapped.x = mapped[0].x;
    closest_mapped.y = mapped[0].y;
    //observations[i].id = closest_mapped.id;
    for (unsigned int j = 1; j < mapped.size(); ++j) {
      if (dist(observations[i].x, observations[i].y, mapped[j].x, mapped[j].y) < dist(observations[i].x, observations[i].y, closest_mapped.x, closest_mapped.y)) {
        closest_mapped.id = mapped[j].id;
        closest_mapped.x = mapped[j].x;
        closest_mapped.y = mapped[j].y;
        //observations[i].id = closest_mapped.id;
      }
    }
    corresponding_landmarks.push_back(closest_mapped);
  }
  return corresponding_landmarks;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
      return ;
  double normaliser = 0.0;

  // Update particle weights using a mult-variate Gaussian distribution
  for (int i = 0; i < num_particles; ++i) {
    // Vector of observations relative to particle[i]
    std::vector<LandmarkObs> particle_observations;

    for (unsigned int j = 0; j < observations.size(); ++j) {
      LandmarkObs p_observation;

      // Transform vehicle observation to map coordinate system for particle[i]
      double obs_map_x = transform_x(observations[j].x, observations[j].y, particles[i].x, particles[i].theta);
      double obs_map_y = transform_y(observations[j].x, observations[j].y, particles[i].y, particles[i].theta);

      //if (dist(observations[i].x, observations[i].y, mapped[j].x, mapped[j].y < sensor_range) {

        // Push particle's perspective of observation into vector
        p_observation.id = observations[j].id;
        p_observation.x = obs_map_x;
        p_observation.y = obs_map_y;
        particle_observations.push_back(p_observation);
      //}
    }

    // Calculate nearest neighbour
    std::vector<LandmarkObs> corresponding_landmarks = dataAssociation(map_landmarks.landmark_list, particle_observations);

    // Update particle's weight
    cout << "Prior Particle weight " << particles[i].weight << endl;
    for (unsigned int j = 0; j < observations.size(); ++j) {
      if (j == 0) {
        cout << "OBS x and y (" << observations[j].x << "," << observations[j].y << ")" << endl;
        cout << "LDM x and y (" << corresponding_landmarks[j].x << "," << corresponding_landmarks[j].y << ")" << endl;
      }
      particles[i].weight *= particle_weight(observations[j].x, observations[j].y, corresponding_landmarks[j].x, corresponding_landmarks[j].y, std_landmark[0], std_landmark[1]);
    }
    cout << "Updated Particle weight " << particles[i].weight << endl;

    if (particles[i].weight > normaliser) {
      normaliser = particles[i].weight;
    }
  }

  // Normalise weights
  for (int i = 0; i < num_particles; ++i) {
    particles[i].weight = particles[i].weight / normaliser;
    weights[i] = particles[i].weight; // used in resample distribution
    cout << "Normalised Particle weight " << particles[i].weight << endl;
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement
  return ;
  default_random_engine gen;
  std::vector<Particle> resampled_particles;

  std::discrete_distribution<int> distribution(weights.begin(), weights.end());

  int p[num_particles]={};
  for (int i = 0; i < num_particles; ++i) {
    int number = distribution(gen);
    ++p[number];
  }

  for (int i = 0; i < num_particles; ++i) {
    for (int j = 0; j < p[i]; ++j) {
      resampled_particles.push_back(particles[i]);
    }
  }

  // update particles
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle,
  const std::vector<int>& associations, const std::vector<double>& sense_x,
  const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle; // TODO added because of return type of function
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
