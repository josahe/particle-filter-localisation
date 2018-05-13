/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *
 * Modified on: May 13, 2018
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

  // Standard deviations for x, y, and theta
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Normal distributions for x, y and theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Number of particles
  num_particles = 200;

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

  // Standard deviations for x, y, and theta
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // Update particle positions using motion model
  for (int i = 0; i < num_particles; ++i) {
    particles[i].x = update_x(particles[i].x, velocity, particles[i].theta,
      yaw_rate, delta_t);
    particles[i].y = update_y(particles[i].y, velocity, particles[i].theta,
      yaw_rate, delta_t);
    particles[i].theta = update_theta(particles[i].theta, yaw_rate, delta_t);

    // Normal distributions for x, y and theta
    normal_distribution<double> dist_x(particles[i].x, std_x);
    normal_distribution<double> dist_y(particles[i].y, std_y);
    normal_distribution<double> dist_theta(particles[i].theta, std_theta);

    // TODO reintroduce when weight update and resampling works
    // Add noise to motion model
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> mapped, int idx) {
  // Update particle observation to nearest neighbor
  for (unsigned int i = 0; i < particles[idx].observations.size(); ++i) {
    LandmarkObs closest_mapped;
    closest_mapped.id = mapped[0].id;
    closest_mapped.x = mapped[0].x;
    closest_mapped.y = mapped[0].y;
    for (unsigned int j = 1; j < mapped.size(); ++j) {
      if (dist(particles[idx].o_sense_x[i], particles[idx].o_sense_y[i], mapped[j].x, mapped[j].y) < dist(particles[idx].o_sense_x[i], particles[idx].o_sense_y[i], closest_mapped.x, closest_mapped.y)) {
        closest_mapped.id = mapped[j].id;
        closest_mapped.x = mapped[j].x;
        closest_mapped.y = mapped[j].y;
      }
    }
    particles[idx].associations.push_back(closest_mapped.id);
    particles[idx].a_sense_x.push_back(closest_mapped.x);
    particles[idx].a_sense_y.push_back(closest_mapped.y);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];

  // Update particle weights using a mult-variate Gaussian distribution
  for (int i = 0; i < num_particles; ++i) {
    // Init particle's weight, observations and associations
    particles[i].weight = 1.0;
    particles[i].observations.clear();
    particles[i].o_sense_x.clear();
    particles[i].o_sense_y.clear();
    particles[i].associations.clear();
    particles[i].a_sense_x.clear();
    particles[i].a_sense_y.clear();

    // Transform each vehicle observation within range to a corresponding map x,y coordinate relative to particle[i]
    for (unsigned int j = 0; j < observations.size(); ++j) {
      // check if vehicle observation is within sensor range
      if (dist(0, 0, observations[j].x, observations[j].y) < sensor_range) {
        // Perform transformations
        particles[i].observations.push_back(observations[j].id);
        particles[i].o_sense_x.push_back(transform_x(observations[j].x, observations[j].y, particles[i].x, particles[i].theta));
        particles[i].o_sense_y.push_back(transform_y(observations[j].x, observations[j].y, particles[i].y, particles[i].theta));
      }
    }

    // Add nearest neighbour for each transformed observation to particle associations
    dataAssociation(map_landmarks.landmark_list, i);

    // Update particle's weight
    for (unsigned int j = 0; j < particles[i].observations.size(); ++j) {
      particles[i].weight *= particle_weight(particles[i].o_sense_x[j], particles[i].o_sense_y[j], particles[i].a_sense_x[j], particles[i].a_sense_y[j], std_x, std_y);
    }
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement
  default_random_engine gen;
  std::vector<Particle> resampled_particles;

  for (int i = 0; i < num_particles; ++i) {
    weights[i] = particles[i].weight;
  }

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

  if (int(resampled_particles.size()) != num_particles) {
    cout << "ERROR: Number of particles is " << resampled_particles.size() << endl;
  }

  // update particles
  particles = resampled_particles;
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
	vector<double> v = best.o_sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.o_sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
