#pragma once
#include "Punto.h"
#include <tuple>

bool controlloCluster(std::vector<Punto> ds, std::vector<Punto> precedente_ds, int numPunti);
//std::tuple<std::vector<Punto>, std::vector<Punto>> sequential_kmeans(std::vector<Punto> ds, std::vector<Punto> centroidi, int k);
std::tuple<std::vector<Punto>, std::vector<Punto>> sequential_kmeans(std::vector<Punto> ds, std::vector<Punto> centroidi, int k, int maxIters, double tol);
