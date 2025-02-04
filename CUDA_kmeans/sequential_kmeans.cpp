#include "sequential_kmeans.h"

#include "sequential_kmeans.h"
#include <cmath>
#include <iostream>
#include <iomanip>

// Funzione per aggiornare le posizioni dei centroidi in base ai punti assegnati
void aggiornamentoCentroidi(std::vector<Punto>& ds, std::vector<Punto>& centroidi, int k, const unsigned long numPunti, const unsigned long dimPunti, std::vector<int>& count) {

    // inizializzo (riempio) il vettore count con zeri; count tiene traccia del numero di punti assegnati a ciascun cluster
    std::fill(count.begin(), count.end(), 0);

    // Imposto le coordinate di tutti i centroidi a zero in preparazione per il calcolo delle nuove posizioni
    for (int j = 0; j < k; j++) {
        std::fill(centroidi[j].dimensioni.begin(), centroidi[j].dimensioni.end(), 0);
    }


    // Itero su tutti i punti del dataset per sommare le coordinate di ciascun punto al centroide corrispondente
    for (int i = 0; i < numPunti; i++) {
        for (int h = 0; h < dimPunti; h++) {
            // Aggiungo il valore della dimensione h del punto i al centroide corrispondente
            centroidi[ds[i].cluster_id].dimensioni[h] += ds[i].dimensioni[h];
        }
        // Incremento il conteggio dei punti assegnati a questo cluster
        count[ds[i].cluster_id]++;
    }


    // Calcolo la media delle coordinate di ciascun centroide per ottenere la nuova posizione
    for (int j = 0; j < k; j++) {
        for (int h = 0; h < dimPunti; h++) {
            // Divido la somma delle coordinate per il numero di punti nel cluster in modo da ottenere la nuova posizione del centroide
            centroidi[j].dimensioni[h] /= count[j];
        }
    }

    /*
    std::cout << "Centroidi aggiornati in aggiornamento centroidi: ";
    for (const auto& c : centroidi) {
        std::cout << "(";
        for (const auto& d : c.dimensioni) {
            std::cout << d << " ";
        }
        std::cout << ") ";
    }
    std::cout << "\n";

    */

}

// Funzione per assegnare ciascun punto del dataset al cluster più vicino basato sui centroidi
void assegnamento(std::vector<Punto>& ds, std::vector<Punto>& centroidi, int k, const unsigned long numPunti, const unsigned long dimPunti) {
    int etichetta_cluster = -1;
    double distanza = std::numeric_limits<double>::max();
    double minDistanza;
    // Itero su ogni punto nel dataset (ds) per assegnarlo al cluster più vicino
    for (int i = 0; i < numPunti; i++) {
        // Inizializzo minDistanza al massimo valore possibile per un double
        // per trovare la minima distanza dal punto corrente a qualsiasi centroide
        minDistanza = std::numeric_limits<double>::max();

        // Itero su tutti i centroidi per trovare il centroide più vicino al punto corrente
        for (int j = 0; j < k; j++) {
            // Inizializzo la distanza a 0 per calcolare la distanza euclidea
            distanza = 0;

            // Calcolo la somma dei quadrati delle differenze tra le coordinate del punto e del centroide
            for (int h = 0; h < dimPunti; h++) {
                distanza += pow(ds[i].dimensioni[h] - centroidi[j].dimensioni[h], 2);
            }

            // Calcolo la radice quadrata della somma per ottenere la distanza euclidea
            distanza = sqrt(distanza);

            // Se la distanza calcolata è minore della minima distanza trovata finora,
            // aggiorno minDistanza e memorizzo l'indice del centroide (j) come il cluster più vicino
            if (distanza < minDistanza) {
                minDistanza = distanza;
                etichetta_cluster = j;
            }
        }

        // Assegno al punto corrente (ds[i]) il cluster id corrispondente al centroide più vicino
        ds[i].cluster_id = etichetta_cluster;

        //std::cout << "Punto " << i << " assegnato al cluster " << etichetta_cluster << " con distanza " << std::setprecision(15) << minDistanza << "\n";

    }

}

// Funzione per verificare se i cluster dei punti sono rimasti invariati tra due iterazioni
/*bool controlloCluster(std::vector<Punto> ds, std::vector<Punto> precedente_ds, int numPunti) {

    // Itera su ogni punto nel dataset per confrontare l'assegnazione del cluster
    for (int i = 0; i < numPunti; i++) {

        // Confronta l'ID del cluster del punto corrente nella versione attuale (ds)
        // con l'ID del cluster del punto corrispondente nella versione precedente (precedente_ds)
        if (ds[i].cluster_id != precedente_ds[i].cluster_id) {
            // Se l'ID del cluster differisce, significa che i cluster sono cambiati
            // quindi ritorna false
            return false;
        }
    }
    return true;
}
*/

bool controlloCluster(const std::vector<Punto>& centroidi, const std::vector<Punto>& centroidi_precedenti, int k, double tol) {
    //std::cout << "Controllo cluster con tolleranza " << tol << "\n";
    for (int i = 0; i < k; i++) {
        double shift = 0.0;
        for (size_t d = 0; d < centroidi[i].dimensioni.size(); d++) {
            double diff = centroidi[i].dimensioni[d] - centroidi_precedenti[i].dimensioni[d];
            shift += diff * diff;
        }
        //std::cout << "Cluster " << i << " shift: " << std::sqrt(shift) << "\n";
        //normalizzo e valuto sulla base della tolleranza
        //normalizzo per adeguare la tolleranza alle dimensioni del dataset in uso
        if ((std::sqrt(shift)/centroidi[i].dimensioni.size()) > tol) {
            //std::cout << "Cluster non ancora convergente.\n";
            return false;
        }
    }
    //std::cout << "Cluster convergente.\n";
    return true;
}

// Funzione per eseguire l'algoritmo K-Means in modo sequenziale
std::tuple<std::vector<Punto>, std::vector<Punto>> sequential_kmeans(std::vector<Punto> ds, std::vector<Punto> centroidi, int k, int maxIter, double tol) {

    int iter = 0; // Contatore per il numero di iterazioni dell'algoritmo
    bool convergenza = false; // Flag per indicare se l'algoritmo è convergente
    bool prima_iter = true; // Flag per indicare se è la prima iterazione
    std::vector<Punto> centroidi_precedenti=centroidi;
    std::vector<Punto> precedente_ds;// Vettore per memorizzare il dataset della iterazione precedente

    // Numero di punti nel dataset (ds)
    const auto numPunti = ds.size();
    // Dimensione di ciascun punto (numero di dimensioni)
    const auto dimPunti = ds[0].dimensioni.size();

    //double distanza, minDistanza; // Variabili per le distanze tra punti e centroidi
    std::vector<int> count(k); // Vettore per contare il numero di punti assegnati a ciascun cluster

    // Ciclo principale dell'algoritmo K-Means, continua finché non si raggiunge la convergenza
    while (!convergenza) {
        //std::cout << "Iterazione: " << iter + 1 << "\n";
        // Trova il centroide più vicino per ogni punto e assegna i punti ai cluster
        assegnamento(ds, centroidi, k, numPunti, dimPunti);

        // Aggiorna le posizioni dei centroidi basandosi sui punti assegnati
        // Inizializza prima i centroidi a zero e poi calcola le nuove posizioni
        aggiornamentoCentroidi(ds, centroidi, k, numPunti, dimPunti, count);  

        // Incrementa il contatore delle iterazioni
        iter++;

        // Verifico se i cluster sono cambiati rispetto all'iterazione precedente
        if ( iter >= maxIter || controlloCluster(centroidi, centroidi_precedenti, k, tol)) {
            convergenza = true;
        }
        else {
            // Aggiorno il dataset precedente con il dataset corrente e imposta prima_iter a false
            /*
            std::cout << "Centroidi precedente: ";
            for (const auto& c : centroidi_precedenti) {
                std::cout << "(";
                for (const auto& d : c.dimensioni) {
                    std::cout << d << " ";
                }
                std::cout << ") ";
            }
            std::cout << "\n";

            */
            centroidi_precedenti = centroidi;
            precedente_ds = ds;

            /*
            std::cout << "Centroidi precedente aggiornato: ";
            for (const auto& c : centroidi_precedenti) {
                std::cout << "(";
                for (const auto& d : c.dimensioni) {
                    std::cout << d << " ";
                }
                std::cout << ") ";
            }
            std::cout << "\n";

            */

        }
    }

    // Stampa il numero totale di iterazioni eseguite
    std::cout << "Numero di iterazioni per convergenza: " << iter << " \n";

    // Ritorna una tupla contenente il dataset aggiornato e i centroidi finali
    return { ds, centroidi };

}

