#pragma once
#include <iostream>
#include <vector>
#include <cmath>

struct Punto {
    std::vector <double> dimensioni; // Coordinate del punto in uno spazio multidimensionale
    int cluster_id{}; // Identificativo del cluster a cui il punto appartiene

    //int id_cluster;                    // Identificativo del cluster a cui il punto appartiene
    //std::vector<int> dimensioni;       // Coordinate del punto in uno spazio multidimensionale

    // Costruttore per inizializzare un punto con un numero di dimensioni specificato
    //Punto(int num_dimensioni) : cluster_id(-1), dimensioni(num_dimensioni, 0) {}

    // Metodo per calcolare la distanza euclidea da un altro punto
    double distanzaDa(const Punto& altro) const {
        double somma = 0.0;
        for (size_t i = 0; i < dimensioni.size(); ++i) {
            double differenza = dimensioni[i] - altro.dimensioni[i];
            somma += differenza * differenza;
        }
        return sqrt(somma);
    }

    // Metodo per stampare le coordinate del punto e l'id del cluster
    void stampa() const {
        std::cout << "Punto (ID Cluster: " << cluster_id << "): [ ";
        for (int dim : dimensioni) {
            std::cout << dim << " ";
        }
        std::cout << "]\n";
    }
};


