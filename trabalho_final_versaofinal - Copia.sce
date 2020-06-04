// Autor: Antonio Mendes M. Junior
// Código: Trabalho Final - Classificação de Vinhos
// Data: 23/08/2013
// Interpretador: Scilab 5.4.1
//===================================================================

clear;// limpar 
clc;

// Obs: Instalar módulo ANN Toobox
//--------------------------------------------------------------------
//                             FUNÇÕES 


//função para converter de binário para decimal
function dec=converte_bin_dec(bin)
    dec=[];
    // varrerá todas as linhas da coluna i em busca do valor 1
    for i=1:size(bin,2)  
        for j=1:size(bin,1) // varrerá todas as linhas
            if bin(j,i)==1 then //se há "1",ele salva o 
                //indice da linha em dec
                dec=[dec j]
                break;  // quando houver dados não LS, ele para o if
                // assim que classificar como sendo de alguma classe    
                else
                //caso n exista '1', classificará aleatoriamente a
                //amostra / se colocar 1 antes, n passará nesse laço.
                if j==size(bin,1) 
                    // gera números inteiros aleatórios
                    // entre 1 e o número de linhas da matriz bin
                    dec=[dec grand(1,1,"uin",1,size(bin,1))];
                end
            end
        end
    end
endfunction


//função para converter de decimal para binário
function bin=converte_dec_bin(dec)
    // bin=zeros(); // matriz de zeros (original)
    bin=[]; //  (atualizado 2020)
    
    for i=1:size(dec,2)   //varre as colunas
          aux=dec(i);    //recebe o elemento que tá na posição j,i
          bin(aux,i)=1; //usa o elemento como indice para inserir "1"
                       //expandindo com zero o restante da matriz
    end
endfunction


//função para misturar os dados
function [xp, cp]=mistura(xin,c)
    cp=[];
    xp=[];
    aux=[1:1:size(xin,2)]

    //grand(1,'prm',aux) retorna uma permutação possível do vetor aux
    temp=grand(1,'prm',aux'); // misturar

    for i = 1:size(xin,2) // laço até o número de amostras        
        xp=[xp,xin(:,temp(i))]; // trocar posição dados
        cp=[cp,c(temp(i))]; // trocar posição da classificação
    end      
endfunction   

//====================================================================

//rand('seed',0);

//----------------------------------------------------------------

// preparação dos dados

// lendos os dados de arq. txt (-1=todas as linhas / 14 = núm. colunas)
dados=read("dados-vinho.txt",-1,14); 
// passando para o formato de amostras por colunas
dados=dados';

// normalizando os dados
//começa na linha 2, pois a 1 é a da classificação das amostras
for i=2:size(dados,1) 
    dados(i,:)=dados(i,:)/max(dados(i,:));
end

// seperando os dados em classes
classe1=dados(:,1:59); // dados da classe 1
classe2=dados(:,60:130); // dados da classe 2
classe3=dados(:,131:178); // dados da classe 3

// amostras de 70% dos dados de cada classe
// essas amostras serão usadas para o treinamento da rede
// e o restante (30%) para teste

// amostras para treinamento (75% dos dados)
ATreclasse1=classe1(:,1:int(size(classe1,2)*0.75));
ATreclasse2=classe2(:,1:int(size(classe2,2)*0.75));
ATreclasse3=classe3(:,1:int(size(classe3,2)*0.75));

// amostras para teste (25% dos dados)
ATestclasse1=classe1(:,(size(ATreclasse1,2)+1):size(classe1,2));
ATestclasse2=classe2(:,(size(ATreclasse2,2)+1):size(classe2,2));
ATestclasse3=classe3(:,(size(ATreclasse3,2)+1):size(classe3,2));


// matriz com os dados para treinamento
dados_treinamento=[ATreclasse1 ATreclasse2 ATreclasse3];

// matriz com os dados para teste
dados_teste=[ATestclasse1 ATestclasse2 ATestclasse3];
c_teste_esperada_dec=dados_teste(1,:);
dados_teste=dados_teste(2:size(dados_teste,1),:);

//----------------------------------------------------------------

// preparação da rede

// recebendo os valores alvo 
c_esperada = dados_treinamento(1,:);
// recebendo os dados de entrada
dados_entrada=dados_treinamento(2:size(dados_treinamento,1),:);


// definição da rede
// neurônios por camada, incluindo entrada
// entrada / camada escondida / saída
N  = [13 7 3];

// parametros de aprendizagem
// taxa de aprendizado / valor minimo atualizar peso
// taxa de momento / valor minimo pra atualizar momento
lp = [0.7 , 0 , 0.3 , 0];

disp('Inicializando a rede...');
W = ann_FF_init(N);

// Delta_W=hypermat(size(W)'); //iniciar Delta_W como 0 (versão original)
Delta_W = repmat(0, size(W)); //iniciar Delta_W como 0 (versão atualizada 2020)

disp('Treinando a rede...');

// quantidade de épocas
T = 300; 
ErroEpoca=[];

for epocas=1:T //treinamento
    // misturar dados para melhor generalização
    [dados_mist,c_mist]=mistura(dados_entrada,c_esperada);
    // convertendo classificação para binário
    c_bin = converte_dec_bin(c_mist);
   
    [W,Delta_W]=ann_FF_Mom_online(dados_mist,c_bin,N,W,lp,1,Delta_W); 
    y=ann_FF_run(dados_mist,N,W);
    e=c_bin-y;
    ErroEpoca=[ErroEpoca  sum(e)^2];
end

y_dec=converte_bin_dec(round(y));
disp(y_dec,'Classicação: '); //apresentar classificação na tela

// Plotando EQM
scf(0); 
plot(ErroEpoca);    
title('EQM com Momento');
xlabel("Épocas");
ylabel("EQM");


//-------------------------------------------------------------
// USANDO OS DADOS DE TESTE


// convertendo classificação dos dados de teste para binário
c_teste_bin=converte_dec_bin(c_teste_esperada_dec); 
// obtendo a classificação para os dados de teste 
// e arredeondando para ficar em binário (0 ou 1)
y_teste_bin=round(ann_FF_run(dados_teste,N,W));

// convertendo a classificação para decimal
y_teste_dec=converte_bin_dec(y_teste_bin);

ErroTeste=0; // o erro de teste se inicia em 0

// teste para verificar se houve alguma classificação errada
// se a classificação for diferente da esperada, será somado +1 
// no valor do ErroTeste
for i=1:size(dados_teste,2)
    if(c_teste_esperada_dec(i) <> y_teste_dec(i))
        ErroTeste=ErroTeste+1;
    end
end

//porcentagem de erros de classificação
porcent_erro=(ErroTeste/size(dados_teste,2))*100

//gráfico pizza de erros x acertos
//[1 1] é o afastamento de cada fatia
//se o ErroTeste = 0 precisamos plotar apenas os acertos
//a função n permite passar 0 como parâmetro
if (ErroTeste==0) then
    scf(1);
    pie([size(dados_teste,2)]);
    legend('Acertos');
else
    scf(1);
    pie([ErroTeste size(dados_teste,2)],[2 2]);
    legend(['Erros';'Acertos']);
end

