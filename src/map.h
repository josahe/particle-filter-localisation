/*
 * map.h
 *
 *  Created on: Dec 12, 2016
 *      Author: mufferm
 */

#ifndef MAP_H_
#define MAP_H_

#include <vector>

class Map {
  public:

  struct single_landmark_s{

  	int id ; // Landmark ID
  	double x; // Landmark x-position in the map (global coordinates)
  	double y; // Landmark y-position in the map (global coordinates)
  };

	std::vector<single_landmark_s> landmark_list; // List of landmarks in the map

};

#endif /* MAP_H_ */
