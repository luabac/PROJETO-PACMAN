# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from multiAgents import MultiAgentSearchAgent


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        """
        Retorna a melhor ação do Pac-Man usando o algoritmo Minimax.

        - Pac-Man (agentIndex == 0) é o nó MAX: escolhe o maior score.
        - Fantasmas (agentIndex > 0) são nós MIN: escolhem o menor score.
        - A profundidade só aumenta depois que o último fantasma joga
          (um "ply" completo = Pac-Man + todos os fantasmas).
        """

        def minimax(agentIndex=0, depth=0, state=gameState):
            # Caso base: jogo acabou ou profundidade máxima atingida.
            # Usa a betterEvaluationFunction (definida abaixo) como valor da folha.
            if state.isWin() or state.isLose() or depth == self.depth:
                return betterEvaluationFunction(state)

            # Próximo agente: volta ao Pac-Man após o último fantasma
            numAgents = state.getNumAgents()
            nextAgent = 0 if agentIndex == numAgents - 1 else agentIndex + 1

            # A profundidade só aumenta quando o último fantasma acabou de jogar
            nextDepth = depth + 1 if agentIndex == numAgents - 1 else depth

            actions = state.getLegalActions(agentIndex)

            # Sem ações legais (agente encurralado): trata como estado terminal
            if not actions:
                return betterEvaluationFunction(state)

            # ---------- Turno do Pac-Man: MAXIMIZAÇÃO ----------
            if agentIndex == self.index:
                bestScore = -float('inf')
                bestAction = None
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgent, nextDepth, successor)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action

                # Na chamada raiz (depth 0 e state original) retorna a ação;
                # nas chamadas recursivas retorna apenas o valor.
                if depth == 0 and state is gameState:
                    return bestAction
                return bestScore

            # ---------- Turno dos fantasmas: MINIMIZAÇÃO ----------
            bestScore = float('inf')
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                score = minimax(nextAgent, nextDepth, successor)
                if score < bestScore:
                    bestScore = score
            return bestScore

        return minimax()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Função de avaliação usada nas folhas do Minimax.

    Combina: score do jogo, distância até a comida mais próxima, quantidade
    de comida/cápsulas restantes, perigo dos fantasmas ativos e atração por
    fantasmas assustados (que podem ser comidos).
    """
    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    # Comida: quanto mais perto da comida mais próxima, melhor;
    # quanto menos comida restante, melhor
    if foodList:
        score += 10.0 / min(manhattanDistance(pos, f) for f in foodList)
    score -= 4.0 * len(foodList)

    # Cápsulas: incentiva pegar (e penaliza as que sobraram)
    if capsules:
        score += 3.0 / min(manhattanDistance(pos, c) for c in capsules)
    score -= 20.0 * len(capsules)

    # Fantasmas
    for ghost in ghostStates:
        dist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            # Fantasma assustado: vale a pena perseguir (se der tempo)
            if dist <= ghost.scaredTimer:
                score += 200.0 / (dist + 1)
        else:
            # Fantasma ativo: perigo cresce muito quando está colado
            if dist <= 1:
                score -= 500.0
            elif dist <= 4:
                score -= 100.0 / (dist * dist)

    return score


# Abbreviation
better = betterEvaluationFunction