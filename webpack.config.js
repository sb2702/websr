const path = require('path');
const webpack = require('webpack');

const CopyWebpackPlugin = require('copy-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');


module.exports = {
    entry: [ "./src/main.ts"],
    output: {
        library: "WebSR",
        libraryTarget: "umd",
        libraryExport: "default",
        path: path.resolve(__dirname, './demo'),
        chunkFilename: 'websr.[name].js',
        filename: "websr.js"
    },
    module: {

        rules: [
            {
                test: /\.ts?$/,
                use: {
                    loader: "ts-loader",
                    options: {
                        compilerOptions:{
                            "types": ["@webgpu/types"]
                        }
                    }
                },
                exclude: /node_modules/,

            },
        ],

    },

    plugins: [

        new HtmlWebpackPlugin({
            template: 'src/demo/index.html'
        }),

        new CleanWebpackPlugin({
            cleanStaleWebpackAssets: false
        }),

        new CopyWebpackPlugin( {
            patterns: [
                { from: "src/demo/*.png", to: path.basename('[name].png') },
                { from: "src/demo/*.mp4", to: path.basename('[name].mp4') },
                { from: "src/demo/*.webm", to: path.basename('[name].webm') },
                { from: "src/demo/*.css", to: path.basename('[name].css') },
                { from: "src/demo/*.js", to: path.basename('[name].js') },
                { from: "src/demo/*.map", to: path.basename('[name].map') },
                { from: "src/demo/*.json", to: path.basename('[name].json') }
            ]
        })

    ],
    resolve: {
        extensions: [".ts", ".tsx", ".js"]
    },

    devServer: {
        static: {
            directory: path.join(__dirname, 'demo'),
        },
        compress: true,
        port: 8000,
    },

    mode: 'development'

};
